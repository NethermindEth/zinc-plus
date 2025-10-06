use super::*;
use crate::{impl_infallible_checked_binary_op, impl_infallible_checked_unary_op};
use ark_ff::{
    AdditiveGroup, CubicExtConfig, CubicExtField, LegendreSymbol, QuadExtConfig, QuadExtField,
    SqrtPrecomputation, fields::Field as ArkWrappedField,
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
    ops::{Add, AddAssign, Deref, Mul, MulAssign, Sub, SubAssign},
};
use num_traits::{
    CheckedAdd, CheckedDiv, CheckedMul, CheckedNeg, CheckedSub, ConstOne, ConstZero, One, Pow, Zero,
};

#[cfg(feature = "rand")]
use ark_std::{UniformRand, rand::prelude::*};
#[cfg(feature = "rand")]
use rand::distr::StandardUniform;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ArkField<F: ArkWrappedField>(F);

impl<F: ArkWrappedField> ArkField<F> {
    /// Wraps a given value into this wrapper type
    #[inline(always)]
    pub const fn new(value: F) -> Self {
        Self(value)
    }

    /// Get the reference to the wrapped value
    #[inline(always)]
    pub const fn inner(&self) -> &F {
        &self.0
    }

    /// Get the wrapped value, consuming self
    #[inline(always)]
    pub const fn into_inner(self) -> F {
        self.0
    }
}

impl<P: QuadExtConfig> ArkField<QuadExtField<P>> {
    pub const fn new_ext(c0: P::BaseField, c1: P::BaseField) -> Self {
        Self(QuadExtField::<P>::new(c0, c1))
    }
}

impl<P: CubicExtConfig> ArkField<CubicExtField<P>> {
    pub const fn new_ext(c0: P::BaseField, c1: P::BaseField, c2: P::BaseField) -> Self {
        Self(CubicExtField::<P>::new(c0, c1, c2))
    }
}

//
// Core traits
//

impl<F: ArkWrappedField> Debug for ArkField<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        Debug::fmt(&self.0, f)
    }
}

impl<F: ArkWrappedField> Display for ArkField<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{}", self.0)
    }
}

impl<F: ArkWrappedField> Default for ArkField<F> {
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

impl<F: ArkWrappedField> Deref for ArkField<F> {
    type Target = F;

    fn deref(&self) -> &Self::Target {
        self.inner()
    }
}

impl<F: ArkWrappedField> PartialOrd for ArkField<F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<F: ArkWrappedField> Ord for ArkField<F> {
    fn cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(&self.0, &other.0)
    }
}

impl<F: ArkWrappedField> Hash for ArkField<F> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

//
// Zero and One traits
//

impl<F: ArkWrappedField> Zero for ArkField<F> {
    #[inline]
    fn zero() -> Self {
        Self(F::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<F: ArkWrappedField> One for ArkField<F> {
    fn one() -> Self {
        Self(F::one())
    }
}

impl<F: ArkWrappedField> ConstZero for ArkField<F> {
    const ZERO: Self = Self(<F as AdditiveGroup>::ZERO);
}

impl<F: ArkWrappedField> ConstOne for ArkField<F> {
    const ONE: Self = Self(F::ONE);
}

//
// Basic arithmetic operations
//

impl<F: ArkWrappedField> Neg for ArkField<F> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(self.0.neg())
    }
}

macro_rules! impl_basic_op {
    ($trait:ident, $method:ident) => {
        impl<F: ArkWrappedField> $trait for ArkField<F> {
            type Output = Self;

            #[inline(always)]
            fn $method(self, rhs: Self) -> Self::Output {
                Self(self.0.$method(rhs.0))
            }
        }

        impl<F: ArkWrappedField> $trait<&Self> for ArkField<F> {
            type Output = Self;

            #[inline(always)]
            fn $method(self, rhs: &Self) -> Self::Output {
                Self(self.0.$method(&rhs.0))
            }
        }

        impl<F: ArkWrappedField> $trait<&mut Self> for ArkField<F> {
            type Output = Self;

            #[inline(always)]
            fn $method(self, rhs: &mut Self) -> Self::Output {
                Self(self.0.$method(&rhs.0))
            }
        }

        impl<F: ArkWrappedField> $trait for &ArkField<F> {
            type Output = ArkField<F>;

            #[inline(always)]
            fn $method(self, rhs: Self) -> Self::Output {
                ArkField(self.0.$method(rhs.0))
            }
        }

        impl<F: ArkWrappedField> $trait<ArkField<F>> for &ArkField<F> {
            type Output = ArkField<F>;

            #[inline(always)]
            fn $method(self, rhs: ArkField<F>) -> Self::Output {
                ArkField(self.0.$method(&rhs.0))
            }
        }
    };
}

impl_basic_op!(Add, add);
impl_basic_op!(Sub, sub);
impl_basic_op!(Mul, mul);

impl<F: ArkWrappedField> Div for ArkField<F> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self.div(&rhs)
    }
}

impl<F: ArkWrappedField> Div<&Self> for ArkField<F> {
    type Output = Self;

    fn div(self, rhs: &Self) -> Self::Output {
        self.checked_div(rhs).expect("Division by zero")
    }
}

impl<F: ArkWrappedField> Div<&mut Self> for ArkField<F> {
    type Output = Self;

    fn div(self, rhs: &mut Self) -> Self::Output {
        self.div(&*rhs)
    }
}

impl<F: ArkWrappedField> Pow<u32> for ArkField<F> {
    type Output = Self;

    fn pow(self, rhs: u32) -> Self::Output {
        Self(self.0.pow([u64::from(rhs)]))
    }
}

impl<F: ArkWrappedField> Inv for ArkField<F> {
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

impl_infallible_checked_unary_op!(ArkField<F: ArkWrappedField>, CheckedNeg, neg);
impl_infallible_checked_binary_op!(ArkField<F: ArkWrappedField>, CheckedAdd, add);
impl_infallible_checked_binary_op!(ArkField<F: ArkWrappedField>, CheckedSub, sub);
impl_infallible_checked_binary_op!(ArkField<F: ArkWrappedField>, CheckedMul, mul);

impl<F: ArkWrappedField> CheckedDiv for ArkField<F> {
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
        impl<F: ArkWrappedField> $trait for ArkField<F> {
            fn $method(&mut self, rhs: Self) {
                // Use reference for inner call to avoid moves of rhs.0 where not needed
                *self = self.$inner(&rhs);
            }
        }

        impl<F: ArkWrappedField> $trait<&Self> for ArkField<F> {
            fn $method(&mut self, rhs: &Self) {
                *self = self.$inner(rhs);
            }
        }

        impl<F: ArkWrappedField> $trait<&mut Self> for ArkField<F> {
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

impl<F: ArkWrappedField> Sum for ArkField<F> {
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl<'a, F: ArkWrappedField> Sum<&'a Self> for ArkField<F> {
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl<F: ArkWrappedField> Product for ArkField<F> {
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

impl<'a, F: ArkWrappedField> Product<&'a Self> for ArkField<F> {
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

//
// Conversions
//

impl<F: ArkWrappedField> From<&ArkField<F>> for ArkField<F> {
    fn from(value: &Self) -> Self {
        *value
    }
}

macro_rules! impl_from_delegate {
    ($($t:ty),* $(,)?) => {
        $(
            impl<F: ArkWrappedField> From<$t> for ArkField<F> {
                fn from(value: $t) -> Self {
                    Self(F::from(value))
                }
            }

            impl<F: ArkWrappedField> From<&$t> for ArkField<F> {
                fn from(value: &$t) -> Self {
                    Self::from(*value)
                }
            }
        )*
    };
}

impl_from_delegate!(u8, u16, u32, u64, u128);
impl_from_delegate!(i8, i16, i32, i64, i128);

impl<F: ArkWrappedField> From<bool> for ArkField<F> {
    fn from(value: bool) -> Self {
        if value { Self::one() } else { Self::zero() }
    }
}

//
// Ring and Field
//

impl<F: ArkWrappedField> Ring for ArkField<F> {}

impl<F: ArkWrappedField> IntRing for ArkField<F> {}

impl<F: ArkWrappedField> Field for ArkField<F> {}

//
// RNG
//

#[cfg(feature = "rand")]
impl<F: ArkWrappedField> Distribution<ArkField<F>> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> ArkField<F> {
        ArkField(UniformRand::rand(rng))
    }
}

#[cfg(feature = "rand")]
impl<F: ArkWrappedField> UniformRand for ArkField<F> {
    fn rand<R: ark_std::rand::Rng + ?Sized>(rng: &mut R) -> Self {
        Self(F::rand(rng))
    }
}

//
// Traits from ark-ff
//

impl<F: ArkWrappedField> CanonicalSerialize for ArkField<F> {
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

impl<F: ArkWrappedField> CanonicalSerializeWithFlags for ArkField<F> {
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

impl<F: ArkWrappedField> CanonicalDeserialize for ArkField<F> {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        F::deserialize_with_mode(reader, compress, validate).map(Self)
    }
}

impl<F: ArkWrappedField> CanonicalDeserializeWithFlags for ArkField<F> {
    fn deserialize_with_flags<R: Read, G: Flags>(
        reader: R,
    ) -> Result<(Self, G), SerializationError> {
        F::deserialize_with_flags(reader).map(|(field, flags)| (Self(field), flags))
    }
}

impl<F: ArkWrappedField> Valid for ArkField<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.0.check()
    }
}

impl<F: ArkWrappedField> AdditiveGroup for ArkField<F> {
    type Scalar = Self;
    const ZERO: Self = <Self as ConstZero>::ZERO;
}

impl<F: ArkWrappedField> zeroize::Zeroize for ArkField<F> {
    fn zeroize(&mut self) {
        self.0.zeroize()
    }
}

impl<F: ArkWrappedField> ArkWrappedField for ArkField<F> {
    type BasePrimeField = F::BasePrimeField;
    const SQRT_PRECOMP: Option<SqrtPrecomputation<Self>> = {
        match F::SQRT_PRECOMP {
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
        F::extension_degree()
    }

    fn to_base_prime_field_elements(&self) -> impl Iterator<Item = Self::BasePrimeField> {
        self.0.to_base_prime_field_elements()
    }

    fn from_base_prime_field_elems(
        elems: impl IntoIterator<Item = Self::BasePrimeField>,
    ) -> Option<Self> {
        F::from_base_prime_field_elems(elems).map(Self)
    }

    fn from_base_prime_field(elem: Self::BasePrimeField) -> Self {
        Self(F::from_base_prime_field(elem))
    }

    fn from_random_bytes_with_flags<G: Flags>(bytes: &[u8]) -> Option<(Self, G)> {
        F::from_random_bytes_with_flags(bytes).map(|(field, flags)| (Self(field), flags))
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
        Self(self.0.mul_by_base_prime_field(elem))
    }
}
