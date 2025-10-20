use super::*;
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
    str::FromStr,
};
use crypto_primitives_proc_macros::InfallibleCheckedOp;
use num_traits::{
    CheckedAdd, CheckedDiv, CheckedMul, CheckedNeg, CheckedSub, ConstOne, ConstZero, One, Pow, Zero,
};

#[cfg(feature = "rand")]
use ark_std::{UniformRand, rand::prelude::*};
#[cfg(feature = "rand")]
use rand::distr::StandardUniform;

#[derive(Clone, Copy, PartialEq, Eq, InfallibleCheckedOp)]
#[infallible_checked_unary_op((CheckedNeg, neg))]
#[infallible_checked_binary_op((CheckedAdd, add), (CheckedSub, sub), (CheckedMul, mul))]
#[repr(transparent)]
pub struct ArkField<F: ArkWrappedField>(F);

impl<F: ArkWrappedField> ArkField<F> {
    /// Wraps a given value into this wrapper type
    #[inline(always)]
    pub const fn new(value: F) -> Self {
        Self(value)
    }

    /// Get the reference to the wrapped value
    #[inline(always)]
    #[must_use]
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

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.inner()
    }
}

impl<F: ArkWrappedField> PartialOrd for ArkField<F> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<F: ArkWrappedField> Ord for ArkField<F> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(&self.0, &other.0)
    }
}

impl<F: ArkWrappedField> Hash for ArkField<F> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

impl<F: ArkWrappedField + FromStr> FromStr for ArkField<F> {
    type Err = <F as FromStr>::Err;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        F::from_str(s).map(Self)
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
    #[inline(always)]
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

    #[inline(always)]
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

    #[inline(always)]
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

    #[inline(always)]
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
        // Pretty much copy the value of `SqrtPrecomputation`
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ConstIntRing, ensure_type_implements_trait};
    use alloc::vec::Vec;
    use ark_ff::{Fp256, MontBackend, MontConfig};
    use num_traits::{One, Zero};

    // Define a test prime field for testing
    // Using a 256-bit prime for testing: 2^256 - 2^32 - 977 (secp256k1 field prime)
    #[derive(MontConfig)]
    #[modulus = "115792089237316195423570985008687907853269984665640564039457584007908834671663"]
    #[generator = "3"]
    pub struct TestFieldConfig;
    type ArkFp = Fp256<MontBackend<TestFieldConfig, 4>>;
    type F = ArkField<ArkFp>;

    #[test]
    fn ensure_blanket_traits() {
        ensure_type_implements_trait!(F, ConstIntRing);
    }

    #[test]
    fn zero_one_basics() {
        let z = F::zero();
        assert!(z.is_zero());
        let o = F::one();
        assert!(!o.is_zero());
        assert_ne!(z, o);
    }

    #[test]
    fn basic_operations() {
        // Negation
        let a: F = 9_u64.into();
        let neg_a = -a;
        assert_eq!(a + neg_a, F::zero());

        let a = F::from(10_u64);
        let b = F::from(5_u64);

        // Addition
        let c = a + b;
        assert_eq!(c, F::from(15_u64));

        // Subtraction
        let d = a - b;
        assert_eq!(d, F::from(5_u64));

        // Multiplication
        let e = a * b;
        assert_eq!(e, F::from(50_u64));

        // Division
        let num: F = 11_u64.into();
        let den: F = 5_u64.into();
        let q = num / den;
        assert_eq!(q * den, num);
    }

    #[test]
    fn add_wrapping() {
        let a: F = (-100_i64).into();
        let b: F = 105_u64.into();
        let c = a + b;
        let d: F = 5_u64.into();
        assert_eq!(c, d);
    }

    #[allow(clippy::op_ref)]
    #[test]
    fn reference_operations() {
        let a = F::from(10_u64);
        let b = F::from(5_u64);

        // Addition
        let c = a + &b;
        assert_eq!(c, F::from(15_u64));

        // Subtraction
        let d = a - &b;
        assert_eq!(d, F::from(5_u64));

        // Multiplication
        let e = a * &b;
        assert_eq!(e, F::from(50_u64));

        // Division
        let num: F = 11_u64.into();
        let den: F = 5_u64.into();
        let q = num / &den;
        assert_eq!(q * &den, num);
    }

    #[test]
    fn from_unsigned_and_signed() {
        assert_eq!(F::from(0_u64), F::zero());
        assert_eq!(F::from(1_u32), F::one());
        assert_eq!(F::from(-1_i32) + F::one(), F::zero());
        assert_eq!(F::from(-5_i64) + F::from(5_u64), F::zero());
    }

    #[test]
    fn from_bool() {
        assert_eq!(F::from(true), F::one());
        assert_eq!(F::from(false), F::zero());

        let t: F = true.into();
        let f: F = false.into();
        assert_eq!(t, F::one());
        assert_eq!(f, F::zero());
    }

    #[test]
    fn assign_operations() {
        // Addition
        let mut a: F = 5_u64.into();
        a += F::from(6_u64);
        assert_eq!(a, 11_u64.into());

        // Subtraction
        let mut a: F = 20_u64.into();
        a -= F::from(7_u64);
        assert_eq!(a, 13_u64.into());

        // Multiplication
        let mut a: F = 11_u64.into();
        a *= F::from(3_u64);
        assert_eq!(a, 33_u64.into());

        // Division
        let mut a: F = 20_u64.into();
        let b: F = 4_u64.into();
        a /= b;
        assert_eq!(a * b, 20_u64.into());
    }

    #[test]
    #[should_panic(expected = "Division by zero")]
    fn div_by_zero_panics() {
        let a: F = 7_u64.into();
        let zero = F::zero();
        let _ = a / zero;
    }

    #[test]
    fn pow_operation() {
        // Test basic exponentiation
        let base = F::from(2_u64);

        // 2^0 = 1
        assert_eq!(base.pow(0), F::one());

        // 2^1 = 2
        assert_eq!(base.pow(1), base);

        // 2^3 = 8
        assert_eq!(base.pow(3), F::from(8_u64));

        // 2^10 = 1024
        assert_eq!(base.pow(10), F::from(1024_u64));

        // Test with different base
        let base = F::from(3_u64);

        // 3^4 = 81
        assert_eq!(base.pow(4), F::from(81_u64));

        // Test with base 1
        let base = F::from(1_u64);
        assert_eq!(base.pow(1000), F::from(1_u64));

        // Test with base 0
        let base = F::from(0_u64);
        assert_eq!(base.pow(0), F::one()); // 0^0 = 1 by convention
        assert_eq!(base.pow(10), F::zero()); // 0^n = 0 for n > 0
    }

    #[test]
    fn inv_operation() {
        let a = F::from(5_u64);
        let inv_a = a.inv().unwrap();
        assert_eq!(a * inv_a, F::one());

        // Test that zero has no inverse
        let zero = F::zero();
        assert!(zero.inv().is_none());
    }

    #[test]
    fn checked_neg() {
        // Test with positive number
        let a = F::from(10_u64);
        let neg_a = a.checked_neg().unwrap();
        assert_eq!(neg_a, F::from(-10_i64));

        // Test with negative number
        let b = F::from(-5_i64);
        let neg_b = b.checked_neg().unwrap();
        assert_eq!(neg_b, F::from(5_u64));

        // Test with zero
        let zero = F::zero();
        let neg_zero = zero.checked_neg().unwrap();
        assert_eq!(neg_zero, zero);
    }

    #[test]
    fn checked_add() {
        let a = F::from(10_u64);
        let b = F::from(5_u64);

        let c = a.checked_add(&b).unwrap();
        assert_eq!(c, F::from(15_u64));
    }

    #[test]
    fn checked_sub() {
        let a = F::from(10_u64);
        let b = F::from(5_u64);

        let d = a.checked_sub(&b).unwrap();
        assert_eq!(d, F::from(5_u64));
    }

    #[test]
    fn checked_mul() {
        let a = F::from(10_u64);
        let b = F::from(5_u64);

        let e = a.checked_mul(&b).unwrap();
        assert_eq!(e, F::from(50_u64));
    }

    #[test]
    fn checked_div() {
        let a = F::from(10_u64);
        let b = F::from(5_u64);
        let zero = F::zero();

        // Normal division
        let c = a.checked_div(&b).unwrap();
        assert_eq!(c * b, a);

        // Division by zero
        assert!(a.checked_div(&zero).is_none());
    }

    #[allow(clippy::op_ref)]
    #[test]
    fn ref_and_value_combinations_add_sub_mul() {
        let a: F = 42_u64.into();
        let b: F = 123_u64.into();

        let r1 = a + b;
        let a1: F = 42_u64.into();
        let b1: F = 123_u64.into();
        let r2 = a1 + b1;
        let r3 = a1 + &b1;
        let a2: F = 42_u64.into();
        let b2: F = 123_u64.into();
        let r4 = &a2 + b2;
        assert_eq!(r1, r2);
        assert_eq!(r1, r3);
        assert_eq!(r1, r4);

        let a: F = 88_u64.into();
        let b: F = 59_u64.into();
        let s1 = a - b;
        let a1: F = 88_u64.into();
        let b1: F = 59_u64.into();
        let s2 = a1 - b1;
        let s3 = a1 - &b1;
        let a2: F = 88_u64.into();
        let b2: F = 59_u64.into();
        let s4 = &a2 - b2;
        assert_eq!(s1, s2);
        assert_eq!(s1, s3);
        assert_eq!(s1, s4);

        let a: F = 9_u64.into();
        let b: F = 14_u64.into();
        let m1 = a * b;
        let a1: F = 9_u64.into();
        let b1: F = 14_u64.into();
        let m2 = a1 * b1;
        let m3 = a1 * &b1;
        let a2: F = 9_u64.into();
        let b2: F = 14_u64.into();
        let m4 = &a2 * b2;
        assert_eq!(m1, m2);
        assert_eq!(m1, m3);
        assert_eq!(m1, m4);
    }

    #[test]
    fn assign_ops_with_refs_and_val() {
        let mut a: F = 100_u64.into();
        let b: F = 50_u64.into();
        a += b;
        assert_eq!(a, 150_u64.into());

        let mut c: F = 100_u64.into();
        let d: F = 50_u64.into();
        c += &d;
        assert_eq!(c, 150_u64.into());

        let mut e: F = 100_u64.into();
        let f: F = 30_u64.into();
        e -= f;
        assert_eq!(e, 70_u64.into());

        let mut g: F = 100_u64.into();
        let h: F = 30_u64.into();
        g -= &h;
        assert_eq!(g, 70_u64.into());

        let mut i: F = 10_u64.into();
        let j: F = 5_u64.into();
        i *= j;
        assert_eq!(i, 50_u64.into());

        let mut k: F = 10_u64.into();
        let l: F = 5_u64.into();
        k *= &l;
        assert_eq!(k, 50_u64.into());
    }

    #[test]
    fn aggregate_operations() {
        // Test Sum trait
        let values: Vec<F> = [1_u64, 2_u64, 3_u64].into_iter().map(F::from).collect();
        let sum: F = values.iter().sum();
        assert_eq!(sum, F::from(6_u64));

        let sum2: F = values.into_iter().sum();
        assert_eq!(sum2, F::from(6_u64));

        // Test Product trait
        let values: Vec<F> = [2_u64, 3_u64, 4_u64].into_iter().map(F::from).collect();
        let product: F = values.iter().product();
        assert_eq!(product, F::from(24_u64));

        let product2: F = values.into_iter().product();
        assert_eq!(product2, F::from(24_u64));

        // Test empty collections
        let empty_vec: Vec<F> = Vec::new();
        let empty_sum: F = empty_vec.iter().sum();
        assert_eq!(empty_sum, F::zero());

        let empty_product: F = empty_vec.iter().product();
        assert_eq!(empty_product, F::one());
    }

    #[test]
    fn conversions() {
        // Test From<ArkWrappedField> for ArkField (via new)
        let inner = ArkFp::from(123_u64);
        let wrapped = F::new(inner);
        assert_eq!(wrapped, F::from(123_u64));

        // Test inner() and into_inner()
        let value = F::from(456_u64);
        let inner_ref = value.inner();
        assert_eq!(*inner_ref, ArkFp::from(456_u64));
        assert_eq!(value.into_inner(), ArkFp::from(456_u64));
    }

    #[test]
    fn from_primitive() {
        // Test from_u8
        let a = F::from(42_u8);
        assert_eq!(a, F::from(42_u64));

        // Test from_u16
        let b = F::from(12345_u16);
        assert_eq!(b, F::from(12345_u64));

        // Test from_u32
        let c = F::from(1234567890_u32);
        assert_eq!(c, F::from(1234567890_u64));

        // Test from_u64
        let d = F::from(1234567890123456789_u64);
        assert_eq!(d, F::from(1234567890123456789_u64));

        // Test from_i8
        let e = F::from(-42_i8);
        assert_eq!(e, F::from(-42_i64));

        // Test from_i16
        let f = F::from(-12345_i16);
        assert_eq!(f, F::from(-12345_i64));

        // Test from_i32
        let g = F::from(-1234567890_i32);
        assert_eq!(g, F::from(-1234567890_i64));

        // Test from_i64
        let h = F::from(-1234567890123456789_i64);
        assert_eq!(h, F::from(-1234567890123456789_i64));
    }

    #[allow(clippy::clone_on_copy)]
    #[test]
    fn clone_and_copy() {
        let a = F::from(42_u64);
        let b = a; // Copy
        let c = a.clone(); // Clone

        assert_eq!(a, b);
        assert_eq!(a, c);
        assert_eq!(b, c);
    }

    #[test]
    fn equality_and_ordering() {
        let a = F::from(10_u64);
        let b = F::from(10_u64);
        let c = F::from(20_u64);

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

        let a = F::from(42_u64);
        let b = F::from(42_u64);

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
        let default_field = F::default();
        assert_eq!(default_field, F::zero());
        assert!(default_field.is_zero());
    }

    #[test]
    fn from_str() {
        // Test parsing from string
        let a: F = "123".parse().unwrap();
        assert_eq!(a, F::from(123_u64));

        let b: F = "0".parse().unwrap();
        assert_eq!(b, F::zero());

        let c: F = "1".parse().unwrap();
        assert_eq!(c, F::one());

        // Fp supports negative numbers
        let d: F = "-123".parse().unwrap();
        assert_eq!(d, F::from(-123_i64));

        // Test invalid cases
        assert!("0x123".parse::<F>().is_err()); // Hex not supported
        assert!("abc".parse::<F>().is_err());
        assert!("12.34".parse::<F>().is_err());
        assert!("".parse::<F>().is_err());
    }

    #[test]
    fn deref_access() {
        let a = F::from(42_u64);
        // Test that we can access inner methods via Deref
        let _ = a.is_zero();
        let _ = a.inverse();
    }
}
