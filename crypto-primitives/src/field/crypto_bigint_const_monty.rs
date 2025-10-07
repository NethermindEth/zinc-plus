use super::*;
use crate::crypto_bigint_int::Int;
use core::{
    cmp::Ordering,
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    hash::{Hash, Hasher},
    iter::{Product, Sum},
    ops::{Add, AddAssign, DivAssign, Mul, MulAssign, Rem, RemAssign, Shl, Shr, Sub, SubAssign},
};
use crypto_bigint::{
    Limb, Uint,
    modular::{ConstMontyForm, ConstMontyParams as Params, Retrieve},
    subtle::{Choice, ConditionallySelectable, ConstantTimeEq},
};
use crypto_primitives_proc_macros::InfallibleCheckedOp;
use num_traits::{
    CheckedAdd, CheckedDiv, CheckedMul, CheckedNeg, CheckedRem, CheckedSub, ConstOne, ConstZero,
    One, Pow, Zero,
};

#[cfg(feature = "rand")]
use rand::{distr::StandardUniform, prelude::*, rand_core::TryRngCore};

#[cfg(target_pointer_width = "64")]
const WORD_FACTOR: usize = 1;
#[cfg(target_pointer_width = "32")]
const WORD_FACTOR: usize = 2;

#[derive(Clone, Copy, PartialEq, Eq, InfallibleCheckedOp)]
#[infallible_checked_unary_op((CheckedNeg, neg))]
#[infallible_checked_binary_op((CheckedAdd, add), (CheckedSub, sub), (CheckedMul, mul))]
#[repr(transparent)]
pub struct ConstMontyField<Mod: Params<LIMBS>, const LIMBS: usize>(ConstMontyForm<Mod, LIMBS>);

impl<Mod: Params<LIMBS>, const LIMBS: usize> ConstMontyField<Mod, LIMBS> {
    pub const LIMBS: usize = Mod::LIMBS;

    #[allow(clippy::cast_possible_truncation)] // Guaranteed to fit due to crypto_bigint::Uint
    pub const BITS: u32 = LIMBS as u32 * Limb::BITS;

    #[inline(always)]
    pub const fn new(value: ConstMontyForm<Mod, LIMBS>) -> Self {
        Self(value)
    }

    #[inline(always)]
    pub const fn inner(&self) -> &ConstMontyForm<Mod, LIMBS> {
        &self.0
    }

    #[inline(always)]
    pub const fn into_inner(self) -> ConstMontyForm<Mod, LIMBS> {
        self.0
    }

    /// Retrieves the integer currently encoded in this [`ConstMontyForm`],
    /// guaranteed to be reduced.
    pub const fn retrieve(&self) -> Uint<LIMBS> {
        self.0.retrieve()
    }

    /// Access the `ConstMontyForm` value in Montgomery form.
    pub const fn as_montgomery(&self) -> &Uint<LIMBS> {
        self.0.as_montgomery()
    }

    /// Mutably access the `ConstMontyForm` value in Montgomery form.
    pub fn as_montgomery_mut(&mut self) -> &mut Uint<LIMBS> {
        self.0.as_montgomery_mut()
    }

    /// Create a `ConstMontyForm` from a value in Montgomery form.
    pub const fn from_montgomery(integer: Uint<LIMBS>) -> Self {
        Self(ConstMontyForm::from_montgomery(integer))
    }

    /// Extract the value from the `ConstMontyForm` in Montgomery form.
    pub const fn to_montgomery(&self) -> Uint<LIMBS> {
        self.0.to_montgomery()
    }

    /// Performs division by 2, that is returns `x` such that `x + x = self`.
    pub const fn div_by_2(&self) -> Self {
        Self(self.0.div_by_2())
    }

    /// Double `self`.
    pub const fn double(&self) -> Self {
        Self(self.0.double())
    }

    /// See [ConstMontyForm::pow_bounded_exp].
    pub const fn pow_bounded_exp<const RHS_LIMBS: usize>(
        &self,
        exponent: &Uint<RHS_LIMBS>,
        exponent_bits: u32,
    ) -> Self {
        Self(self.0.pow_bounded_exp(exponent, exponent_bits))
    }
}

//
// Core traits
//

impl<Mod: Params<LIMBS>, const LIMBS: usize> Debug for ConstMontyField<Mod, LIMBS> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        Debug::fmt(&self.0, f)
    }
}

impl<Mod: Params<LIMBS>, const LIMBS: usize> Display for ConstMontyField<Mod, LIMBS> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{} (mod {})", self.0.retrieve(), Mod::PARAMS.modulus())
    }
}

impl<Mod: Params<LIMBS>, const LIMBS: usize> Default for ConstMontyField<Mod, LIMBS> {
    #[inline(always)]
    fn default() -> Self {
        Self::ZERO
    }
}

impl<Mod: Params<LIMBS>, const LIMBS: usize> PartialOrd for ConstMontyField<Mod, LIMBS> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<Mod: Params<LIMBS>, const LIMBS: usize> Ord for ConstMontyField<Mod, LIMBS> {
    fn cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(self.0.as_montgomery(), other.0.as_montgomery())
    }
}

impl<Mod: Params<LIMBS>, const LIMBS: usize> Hash for ConstMontyField<Mod, LIMBS> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.as_montgomery().hash(state)
    }
}

//
// Zero and One traits
//

impl<Mod: Params<LIMBS>, const LIMBS: usize> Zero for ConstMontyField<Mod, LIMBS> {
    #[inline]
    fn zero() -> Self {
        Self::ZERO
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self == &Self::ZERO
    }
}

impl<Mod: Params<LIMBS>, const LIMBS: usize> One for ConstMontyField<Mod, LIMBS> {
    fn one() -> Self {
        Self::ONE
    }
}

impl<Mod: Params<LIMBS>, const LIMBS: usize> ConstZero for ConstMontyField<Mod, LIMBS> {
    const ZERO: Self = Self(ConstMontyForm::ZERO);
}

impl<Mod: Params<LIMBS>, const LIMBS: usize> ConstOne for ConstMontyField<Mod, LIMBS> {
    const ONE: Self = Self(ConstMontyForm::ONE);
}

//
// Basic arithmetic operations
//

impl<Mod: Params<LIMBS>, const LIMBS: usize> Neg for ConstMontyField<Mod, LIMBS> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(self.0.neg())
    }
}

macro_rules! impl_basic_op {
    ($trait:ident, $method:ident) => {
        impl<Mod: Params<LIMBS>, const LIMBS: usize> $trait for ConstMontyField<Mod, LIMBS> {
            type Output = Self;

            #[inline(always)]
            fn $method(self, rhs: Self) -> Self::Output {
                Self(self.0.$method(rhs.0))
            }
        }

        impl<Mod: Params<LIMBS>, const LIMBS: usize> $trait<&Self> for ConstMontyField<Mod, LIMBS> {
            type Output = Self;

            #[inline(always)]
            fn $method(self, rhs: &Self) -> Self::Output {
                Self(self.0.$method(&rhs.0))
            }
        }

        impl<Mod: Params<LIMBS>, const LIMBS: usize> $trait for &ConstMontyField<Mod, LIMBS> {
            type Output = ConstMontyField<Mod, LIMBS>;

            #[inline(always)]
            fn $method(self, rhs: Self) -> Self::Output {
                ConstMontyField(self.0.$method(rhs.0))
            }
        }

        impl<Mod: Params<LIMBS>, const LIMBS: usize> $trait<ConstMontyField<Mod, LIMBS>>
            for &ConstMontyField<Mod, LIMBS>
        {
            type Output = ConstMontyField<Mod, LIMBS>;

            #[inline(always)]
            fn $method(self, rhs: ConstMontyField<Mod, LIMBS>) -> Self::Output {
                ConstMontyField(self.0.$method(&rhs.0))
            }
        }
    };
}

impl_basic_op!(Add, add);
impl_basic_op!(Sub, sub);
impl_basic_op!(Mul, mul);

impl<Mod: Params<LIMBS>, const LIMBS: usize> Div for ConstMontyField<Mod, LIMBS> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self.div(&rhs)
    }
}

impl<Mod: Params<LIMBS>, const LIMBS: usize> Div<&Self> for ConstMontyField<Mod, LIMBS> {
    type Output = Self;

    fn div(self, rhs: &Self) -> Self::Output {
        self.checked_div(rhs).expect("Division by zero")
    }
}

impl<Mod: Params<LIMBS>, const LIMBS: usize> Rem for ConstMontyField<Mod, LIMBS> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        Self(ConstMontyForm::new(
            &self
                .0
                .retrieve()
                .checked_rem(&rhs.0.retrieve())
                .expect("Division by zero"),
        ))
    }
}

impl<Mod: Params<LIMBS>, const LIMBS: usize> Rem<&Self> for ConstMontyField<Mod, LIMBS> {
    type Output = Self;

    fn rem(self, rhs: &Self) -> Self::Output {
        self.rem(*rhs)
    }
}

impl<Mod: Params<LIMBS>, const LIMBS: usize> Shl<u32> for ConstMontyField<Mod, LIMBS> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    fn shl(mut self, rhs: u32) -> Self::Output {
        let mut value = self.0.retrieve();
        value <<= rhs;
        self.0 = ConstMontyForm::new(&value);
        self
    }
}

impl<Mod: Params<LIMBS>, const LIMBS: usize> Shr<u32> for ConstMontyField<Mod, LIMBS> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    fn shr(mut self, rhs: u32) -> Self::Output {
        let mut value = self.0.retrieve();
        value >>= rhs;
        self.0 = ConstMontyForm::new(&value);
        self
    }
}

impl<Mod: Params<LIMBS>, const LIMBS: usize> Pow<u32> for ConstMontyField<Mod, LIMBS> {
    type Output = Self;

    fn pow(self, rhs: u32) -> Self::Output {
        Self(self.0.pow(&crypto_bigint::U64::from_u32(rhs)))
    }
}

impl<Mod: Params<LIMBS>, const LIMBS: usize> Inv for ConstMontyField<Mod, LIMBS> {
    type Output = Option<Self>;

    fn inv(self) -> Self::Output {
        let result = self.0.invert_vartime();
        if result.is_some().into() {
            Some(Self(result.unwrap()))
        } else {
            None
        }
    }
}

//
// Checked arithmetic operations
// (Note: Field operations do not overflow)
//

impl<Mod: Params<LIMBS>, const LIMBS: usize> CheckedDiv for ConstMontyField<Mod, LIMBS> {
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn checked_div(&self, rhs: &Self) -> Option<Self> {
        let inv = rhs.0.invert();
        if inv.is_none().into() {
            return None; // Division by zero
        }
        // Safe to unwrap since we checked for None above
        let inv = inv.unwrap();
        Some(Self(self.0 * inv))
    }
}

impl<Mod: Params<LIMBS>, const LIMBS: usize> CheckedRem for ConstMontyField<Mod, LIMBS> {
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn checked_rem(&self, v: &Self) -> Option<Self> {
        if v.is_zero() { None } else { Some(*self % v) }
    }
}

//
// Arithmetic assign operations
//

macro_rules! impl_field_op_assign {
    ($trait:ident, $method:ident, $inner:ident) => {
        impl<Mod: Params<LIMBS>, const LIMBS: usize> $trait for ConstMontyField<Mod, LIMBS> {
            fn $method(&mut self, rhs: Self) {
                // Use reference for inner call to avoid moves of rhs.0 where not needed
                *self = self.$inner(&rhs);
            }
        }
        impl<Mod: Params<LIMBS>, const LIMBS: usize> $trait<&Self> for ConstMontyField<Mod, LIMBS> {
            fn $method(&mut self, rhs: &Self) {
                *self = self.$inner(rhs);
            }
        }
    };
}

impl_field_op_assign!(AddAssign, add_assign, add);
impl_field_op_assign!(SubAssign, sub_assign, sub);
impl_field_op_assign!(MulAssign, mul_assign, mul);
impl_field_op_assign!(DivAssign, div_assign, div);
impl_field_op_assign!(RemAssign, rem_assign, rem);

//
// Aggregate operations
//

impl<Mod: Params<LIMBS>, const LIMBS: usize> Sum for ConstMontyField<Mod, LIMBS> {
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl<'a, Mod: Params<LIMBS>, const LIMBS: usize> Sum<&'a Self> for ConstMontyField<Mod, LIMBS> {
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl<Mod: Params<LIMBS>, const LIMBS: usize> Product for ConstMontyField<Mod, LIMBS> {
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}

impl<'a, Mod: Params<LIMBS>, const LIMBS: usize> Product<&'a Self> for ConstMontyField<Mod, LIMBS> {
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}

//
// Conversions
//

impl<Mod: Params<LIMBS>, const LIMBS: usize> From<ConstMontyForm<Mod, LIMBS>>
    for ConstMontyField<Mod, LIMBS>
{
    #[inline(always)]
    fn from(value: ConstMontyForm<Mod, LIMBS>) -> Self {
        Self(value)
    }
}

impl<Mod: Params<LIMBS>, const LIMBS: usize> From<ConstMontyField<Mod, LIMBS>>
    for ConstMontyForm<Mod, LIMBS>
{
    #[inline(always)]
    fn from(value: ConstMontyField<Mod, LIMBS>) -> Self {
        value.0
    }
}

impl<Mod: Params<LIMBS>, const LIMBS: usize> From<&ConstMontyField<Mod, LIMBS>>
    for ConstMontyField<Mod, LIMBS>
{
    fn from(value: &Self) -> Self {
        *value
    }
}

// Macro to implement From for unsigned integer primitives
macro_rules! impl_from_unsigned {
    ($($t:ty),* $(,)?) => {
        $(
            impl<Mod: Params<LIMBS>, const LIMBS: usize> From<$t> for ConstMontyField<Mod, LIMBS> {
                fn from(value: $t) -> Self {
                    let value = Uint::from(value);
                    Self(ConstMontyForm::new(&value))
                }
            }

            impl<Mod: Params<LIMBS>, const LIMBS: usize> From<&$t> for ConstMontyField<Mod, LIMBS> {
                fn from(value: &$t) -> Self {
                    Self::from(*value)
                }
            }
        )*
    };
}

// Macro to implement From for signed integer primitives
macro_rules! impl_from_signed {
    ($($t:ty),* $(,)?) => {
        $(
            impl<Mod: Params<LIMBS>, const LIMBS: usize> From<$t> for ConstMontyField<Mod, LIMBS> {
                #![allow(clippy::arithmetic_side_effects)]
                fn from(value: $t) -> Self {
                    let magnitude = Uint::from(value.abs_diff(0));
                    let form = ConstMontyForm::new(&magnitude);
                    Self(if value.is_negative() { -form } else { form })
                }
            }

            impl<Mod: Params<LIMBS>, const LIMBS: usize> From<&$t> for ConstMontyField<Mod, LIMBS> {
                fn from(value: &$t) -> Self {
                    Self::from(*value)
                }
            }
        )*
    };
}

impl_from_unsigned!(u8, u16, u32, u64, u128);
impl_from_signed!(i8, i16, i32, i64, i128);

impl<Mod: Params<LIMBS>, const LIMBS: usize> From<bool> for ConstMontyField<Mod, LIMBS> {
    fn from(value: bool) -> Self {
        if value { Self::ONE } else { Self::ZERO }
    }
}

impl<Mod: Params<LIMBS>, const LIMBS: usize> From<Uint<LIMBS>> for ConstMontyField<Mod, LIMBS> {
    fn from(value: Uint<LIMBS>) -> Self {
        Self::from(&value)
    }
}

impl<Mod: Params<LIMBS>, const LIMBS: usize> From<&Uint<LIMBS>> for ConstMontyField<Mod, LIMBS> {
    fn from(value: &Uint<LIMBS>) -> Self {
        Self(ConstMontyForm::new(value))
    }
}

impl<Mod: Params<LIMBS>, const LIMBS: usize, const LIMBS2: usize> From<Int<LIMBS2>>
    for ConstMontyField<Mod, LIMBS>
{
    fn from(value: Int<LIMBS2>) -> Self {
        Self::from(value.inner())
    }
}

impl<Mod: Params<LIMBS>, const LIMBS: usize, const LIMBS2: usize> From<&Int<LIMBS2>>
    for ConstMontyField<Mod, LIMBS>
{
    fn from(value: &Int<LIMBS2>) -> Self {
        Self::from(value.inner())
    }
}

impl<Mod: Params<LIMBS>, const LIMBS: usize, const LIMBS2: usize> From<crypto_bigint::Int<LIMBS2>>
    for ConstMontyField<Mod, LIMBS>
{
    fn from(value: crypto_bigint::Int<LIMBS2>) -> Self {
        Self::from(&value)
    }
}

impl<Mod: Params<LIMBS>, const LIMBS: usize, const LIMBS2: usize> From<&crypto_bigint::Int<LIMBS2>>
    for ConstMontyField<Mod, LIMBS>
{
    #![allow(clippy::arithmetic_side_effects)] // False alert
    fn from(value: &crypto_bigint::Int<LIMBS2>) -> Self {
        assert!(
            LIMBS >= LIMBS2,
            "Cannot convert Int with more limbs than ConstMontyField"
        );
        let value = value.resize();
        let result = Self(ConstMontyForm::new(&value.abs()));

        if value.is_negative().into() {
            -result
        } else {
            result
        }
    }
}

//
// Ring and Field
//

impl<Mod: Params<LIMBS>, const LIMBS: usize> Ring for ConstMontyField<Mod, LIMBS> {}

impl<Mod: Params<LIMBS>, const LIMBS: usize> IntRing for ConstMontyField<Mod, LIMBS> {}

impl<Mod: Params<LIMBS>, const LIMBS: usize> Field for ConstMontyField<Mod, LIMBS> {}

impl<Mod: Params<LIMBS>, const LIMBS: usize> PrimeField for ConstMontyField<Mod, LIMBS> {
    type Inner = ConstMontyForm<Mod, LIMBS>;
    const MODULUS: Self::Inner = ConstMontyForm::<Mod, LIMBS>::new(Mod::PARAMS.modulus().as_ref());

    #[inline(always)]
    fn new_unchecked(value: Self::Inner) -> Self {
        Self(value)
    }

    #[inline(always)]
    fn inner(&self) -> &Self::Inner {
        &self.0
    }
}

//
// RNG
//

#[cfg(feature = "rand")]
impl<Mod: Params<LIMBS>, const LIMBS: usize> Distribution<ConstMontyField<Mod, LIMBS>>
    for StandardUniform
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> ConstMontyField<Mod, LIMBS> {
        crypto_bigint::Random::random(rng)
    }
}

#[cfg(feature = "rand")]
impl<Mod: Params<LIMBS>, const LIMBS: usize> crypto_bigint::Random for ConstMontyField<Mod, LIMBS> {
    fn try_random<R: TryRngCore + ?Sized>(rng: &mut R) -> Result<Self, R::Error> {
        ConstMontyForm::try_random(rng).map(Self)
    }
}

//
// Serialization and Deserialization
//

#[cfg(feature = "serde")]
impl<'de, Mod, const LIMBS: usize> serde::Deserialize<'de> for ConstMontyField<Mod, LIMBS>
where
    Mod: Params<LIMBS>,
    Uint<LIMBS>: crypto_bigint::Encoding,
{
    #[inline(always)]
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        ConstMontyForm::<Mod, LIMBS>::deserialize(deserializer).map(Self)
    }
}

#[cfg(feature = "serde")]
impl<Mod, const LIMBS: usize> serde::Serialize for ConstMontyField<Mod, LIMBS>
where
    Mod: Params<LIMBS>,
    Uint<LIMBS>: crypto_bigint::Encoding,
{
    #[inline(always)]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

//
// Traits from crypto_bigint
//

impl<Mod: Params<LIMBS>, const LIMBS: usize> ConstantTimeEq for ConstMontyField<Mod, LIMBS> {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.0.ct_eq(&other.0)
    }
}

impl<Mod: Params<LIMBS>, const LIMBS: usize> ConditionallySelectable
    for ConstMontyField<Mod, LIMBS>
{
    #[inline(always)]
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Self(ConstMontyForm::conditional_select(
            a.inner(),
            b.inner(),
            choice,
        ))
    }
}

impl<Mod: Params<LIMBS>, const LIMBS: usize> crypto_bigint::Zero for ConstMontyField<Mod, LIMBS> {
    #[inline(always)]
    fn zero() -> Self {
        Self::ZERO
    }
}

impl<Mod: Params<LIMBS>, const LIMBS: usize> crypto_bigint::One for ConstMontyField<Mod, LIMBS> {
    #[inline(always)]
    fn one() -> Self {
        Self::ONE
    }
}

impl<Mod: Params<LIMBS>, const LIMBS: usize> crypto_bigint::Square for ConstMontyField<Mod, LIMBS> {
    fn square(&self) -> Self {
        Self(self.0.square())
    }
}

impl<Mod: Params<LIMBS>, const LIMBS: usize> Retrieve for ConstMontyField<Mod, LIMBS> {
    type Output = <ConstMontyForm<Mod, LIMBS> as Retrieve>::Output;
    fn retrieve(&self) -> Self::Output {
        self.retrieve()
    }
}

#[cfg(feature = "zeroize")]
impl<Mod: Params<LIMBS>, const LIMBS: usize> zeroize::DefaultIsZeroes
    for ConstMontyField<Mod, LIMBS>
{
}

//
// Predefined fields of various sizes for convenience
//

pub type F64<Mod> = ConstMontyField<Mod, { crypto_bigint::U64::LIMBS }>;
pub type F128<Mod> = ConstMontyField<Mod, { 2 * WORD_FACTOR }>;
pub type F192<Mod> = ConstMontyField<Mod, { 3 * WORD_FACTOR }>;
pub type F256<Mod> = ConstMontyField<Mod, { 4 * WORD_FACTOR }>;
pub type F320<Mod> = ConstMontyField<Mod, { 5 * WORD_FACTOR }>;
pub type F384<Mod> = ConstMontyField<Mod, { 6 * WORD_FACTOR }>;
pub type F448<Mod> = ConstMontyField<Mod, { 7 * WORD_FACTOR }>;
pub type F512<Mod> = ConstMontyField<Mod, { 8 * WORD_FACTOR }>;
pub type F576<Mod> = ConstMontyField<Mod, { 9 * WORD_FACTOR }>;
pub type F640<Mod> = ConstMontyField<Mod, { 10 * WORD_FACTOR }>;
pub type F704<Mod> = ConstMontyField<Mod, { 11 * WORD_FACTOR }>;
pub type F768<Mod> = ConstMontyField<Mod, { 12 * WORD_FACTOR }>;
pub type F832<Mod> = ConstMontyField<Mod, { 13 * WORD_FACTOR }>;
pub type F896<Mod> = ConstMontyField<Mod, { 14 * WORD_FACTOR }>;
pub type F960<Mod> = ConstMontyField<Mod, { 15 * WORD_FACTOR }>;
pub type F1024<Mod> = ConstMontyField<Mod, { 16 * WORD_FACTOR }>;
pub type F1280<Mod> = ConstMontyField<Mod, { 20 * WORD_FACTOR }>;
pub type F1536<Mod> = ConstMontyField<Mod, { 24 * WORD_FACTOR }>;
pub type F1792<Mod> = ConstMontyField<Mod, { 28 * WORD_FACTOR }>;
pub type F2048<Mod> = ConstMontyField<Mod, { 32 * WORD_FACTOR }>;
pub type F3072<Mod> = ConstMontyField<Mod, { 48 * WORD_FACTOR }>;
pub type F4096<Mod> = ConstMontyField<Mod, { 64 * WORD_FACTOR }>;
pub type F6144<Mod> = ConstMontyField<Mod, { 96 * WORD_FACTOR }>;
pub type F8192<Mod> = ConstMontyField<Mod, { 128 * WORD_FACTOR }>;
pub type F16384<Mod> = ConstMontyField<Mod, { 256 * WORD_FACTOR }>;
pub type F32768<Mod> = ConstMontyField<Mod, { 512 * WORD_FACTOR }>;

#[cfg(test)]
mod tests {
    use crypto_bigint::{Square, U256, const_monty_params};
    use num_traits::{One, Zero};

    use super::*;

    const_monty_params!(
        ModP,
        U256,
        "00dca94d8a1ecce3b6e8755d8999787d0524d8ca1ea755e7af84fb646fa31f27"
    );
    type F = ConstMontyField<ModP, { U256::LIMBS }>;

    #[test]
    fn zero_one_basics() {
        let z = F::zero();
        assert!(z.is_zero());
        let o = F::one();
        assert!(!o.is_zero());
        assert_ne!(z, o);
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
    fn from_uint() {
        let u: Uint<{ U256::LIMBS }> = Uint::from(123_u64);
        let f: F = u.into();
        assert_eq!(f, F::from(123_u64));
    }

    #[test]
    fn basic_add_smoke() {
        let a: F = 123_u64.into();
        let b: F = 456_u64.into();
        assert_eq!(a + b, F::from(579_u64));
    }

    #[test]
    fn add_wrapping_and_basic() {
        let a: F = (-100_i64).into();
        let b: F = 105_u64.into();
        let c = a + b;
        let d: F = 5_u64.into();
        assert_eq!(c, d);
    }

    #[test]
    fn sub_basic() {
        let a: F = 100_u64.into();
        let b: F = 7_u64.into();
        assert_eq!(a - b, 93_u64.into());
    }

    #[test]
    fn mul_basic() {
        let a: F = 100_u64.into();
        let b: F = 7_u64.into();
        assert_eq!(a * b, 700_u64.into());
    }

    #[test]
    fn add_assign_basic() {
        let mut a: F = 5_u64.into();
        a += F::from(6_u64);
        assert_eq!(a, 11_u64.into());
    }

    #[test]
    fn mul_assign_basic() {
        let mut a: F = 11_u64.into();
        a *= F::from(3_u64);
        assert_eq!(a, 33_u64.into());
    }

    #[test]
    fn neg_basic() {
        let a: F = 9_u64.into();
        let neg_a = -a;

        assert_eq!(a + neg_a, F::zero());
    }

    #[test]
    fn div_basic() {
        let num: F = 11_u64.into();
        let den: F = 5_u64.into();
        let q = num / den;
        assert_eq!(q * den, num);
    }

    #[test]
    #[should_panic]
    fn div_by_zero_returns_panics() {
        let a: F = 7_u64.into();
        let zero = F::zero();
        let _ = a / zero;
    }

    #[test]
    fn ref_and_value_combinations_add_sub_mul() {
        let a: F = 42_u64.into();
        let b: F = 123_u64.into();

        let r1 = a + b;
        let a1: F = 42_u64.into();
        let b1: F = 123_u64.into();
        let r2 = a1 + b1;
        let r3 = a1 + b1;
        let a2: F = 42_u64.into();
        let b2: F = 123_u64.into();
        let r4 = a2 + b2;
        assert_eq!(r1, r2);
        assert_eq!(r1, r3);
        assert_eq!(r1, r4);

        let a: F = 88_u64.into();
        let b: F = 59_u64.into();
        let s1 = a - b;
        let a1: F = 88_u64.into();
        let b1: F = 59_u64.into();
        let s2 = a1 - b1;
        let s3 = a1 - b1;
        let a2: F = 88_u64.into();
        let b2: F = 59_u64.into();
        let s4 = a2 - b2;
        assert_eq!(s1, s2);
        assert_eq!(s1, s3);
        assert_eq!(s1, s4);

        let a: F = 9_u64.into();
        let b: F = 14_u64.into();
        let m1 = a * b;
        let a1: F = 9_u64.into();
        let b1: F = 14_u64.into();
        let m2 = a1 * b1;
        let m3 = a1 * b1;
        let a2: F = 9_u64.into();
        let b2: F = 14_u64.into();
        let m4 = a2 * b2;
        assert_eq!(m1, m2);
        assert_eq!(m1, m3);
        assert_eq!(m1, m4);
    }

    #[test]
    fn assign_ops_with_refs_and_values() {
        let mut x: F = 7_u64.into();
        let y: F = 8_u64.into();
        x += y;
        assert_eq!(x, 15_u64.into());
        let mut x: F = 7_u64.into();
        let y: F = 8_u64.into();
        x.add_assign(&y);
        assert_eq!(x, 15_u64.into());
        let mut x: F = 20_u64.into();
        let y: F = 6_u64.into();
        x -= y;
        assert_eq!(x, 14_u64.into());
        let mut x: F = 20_u64.into();
        let y: F = 6_u64.into();
        x.sub_assign(&y);
        assert_eq!(x, 14_u64.into());
        let mut x: F = 5_u64.into();
        let y: F = 9_u64.into();
        x *= y;
        assert_eq!(x, 45_u64.into());
        let mut x: F = 5_u64.into();
        let y: F = 9_u64.into();
        x.mul_assign(&y);
        assert_eq!(x, 45_u64.into());
    }

    #[test]
    fn negation_properties() {
        let a: F = 12345_u64.into();
        let zero = F::zero();
        assert_eq!(a + (-a), zero);
        assert_eq!(-(-a), a);
        assert_eq!(-zero, zero);
    }

    #[test]
    fn inversion_properties() {
        let a: F = 777_u64.into();
        let inv = a.inv().expect("a should be invertible (non-zero)");
        assert_eq!(a * inv, F::one());
        let zero = F::zero();
        assert!(zero.inv().is_none());
    }

    #[test]
    #[should_panic]
    fn division_properties_and_errors() {
        let a: F = 9876_u64.into();
        let b: F = 543_u64.into();
        let q = a / b;
        assert_eq!(q * b, a);
        let c: F = 17_u64.into();
        let bc = b * c;
        let left = a / bc;
        let right = (a / b) / c;
        assert_eq!(left, right);
        let _ = a / F::zero();
    }

    #[test]
    fn ring_identities() {
        let a: F = 3_u64.into();
        let b: F = 5_u64.into();
        let c: F = 7_u64.into();
        assert_eq!(a + F::zero(), a);
        assert_eq!(a * F::one(), a);
        assert_eq!(a * F::zero(), F::zero());
        assert_eq!(a + b, b + a);
        assert_eq!(a * b, b * a);
        assert_eq!((a + b) + c, a + (b + c));
        assert_eq!((a * b) * c, a * (b * c));
        assert_eq!(a * (b + c), a * b + a * c);
        assert_eq!((a + b) * c, a * c + b * c);
        assert_eq!(a - a, F::zero());
    }

    #[test]
    fn sum_and_product_trait_basic() {
        let v = [1_u64.into(), 2_u64.into(), 3_u64.into(), 4_u64.into()];
        let sum1: F = v.iter().cloned().sum();
        let sum2: F = v.iter().sum();
        let sum3: F = v.into_iter().sum();
        let expected_sum: F = 10_u64.into();
        assert_eq!(sum1, expected_sum);
        assert_eq!(sum2, expected_sum);
        assert_eq!(sum3, expected_sum);

        let prod1: F = v.iter().product();
        // owned-product is not implemented; verify an equivalent fold
        let prod2: F = v.iter().fold(F::one(), |acc, x| acc * x);
        let prod3: F = v.into_iter().product();
        let expected_prod: F = (2 * 3 * 4).into();
        assert_eq!(prod1, expected_prod);
        assert_eq!(prod2, expected_prod);
        assert_eq!(prod3, expected_prod);

        // empty iterators: define behavior as neutral elements
        let empty: [F; 0] = [];
        let sum_empty: F = empty.iter().cloned().sum();
        assert_eq!(sum_empty, F::zero());
        let prod_empty: F = empty.iter().product();
        assert_eq!(prod_empty, F::one());
    }

    #[test]
    fn const_time_eq_and_order() {
        let a: F = 10_u64.into();
        let b: F = 10_u64.into();
        let c: F = 11_u64.into();
        assert_eq!(a.ct_eq(&b).unwrap_u8(), 1);
        assert_eq!(a.ct_eq(&c).unwrap_u8(), 0);
        assert!(a.partial_cmp(&c).is_some());
    }

    #[test]
    fn field_methods() {
        let a: F = 10_u64.into();
        let b: F = 5_u64.into();

        // Test const methods
        assert_eq!(a.add(&b), F::from(15_u64));
        assert_eq!(a.sub(&b), F::from(5_u64));
        assert_eq!(a.mul(&b), F::from(50_u64));
        assert_eq!(a.neg(), -a);

        // Test double
        assert_eq!(a.double(), a + a);

        // Test square
        assert_eq!(a.square(), a * a);

        // Test div_by_2
        let even: F = 20_u64.into();
        let half = even.div_by_2();
        assert_eq!(half + half, even);
    }

    #[test]
    fn wrapper_methods() {
        let value = ConstMontyForm::new(&Uint::from(42_u64));
        let field = F::new(value);

        // Test inner and into_inner
        assert_eq!(field.inner(), &value);
        assert_eq!(field.into_inner(), value);

        // Test retrieve
        let a: F = 123_u64.into();
        let retrieved = a.retrieve();
        assert_eq!(retrieved, Uint::from(123_u64));

        // Test montgomery form conversions
        let b: F = 456_u64.into();
        let mont = b.to_montgomery();
        let from_mont = F::from_montgomery(mont);
        assert_eq!(from_mont, b);

        // Test as_montgomery
        let c: F = 789_u64.into();
        let mont_ref = c.as_montgomery();
        assert_eq!(*mont_ref, c.to_montgomery());

        // Test as_montgomery_mut
        let mut d: F = 100_u64.into();
        let original_mont = d.to_montgomery();
        let mont_mut = d.as_montgomery_mut();
        assert_eq!(*mont_mut, original_mont);
    }

    #[test]
    fn pow_bounded_exp_method() {
        let base: F = 2_u64.into();
        let exponent = Uint::<4>::from(10_u64);
        let result = base.pow_bounded_exp(&exponent, 64);
        assert_eq!(result, F::from(1024_u64));
    }

    #[test]
    fn formatting_traits() {
        let a: F = 255_u64.into();

        // Test Debug - should contain some representation
        let debug_str = alloc::format!("{:?}", a);
        assert!(!debug_str.is_empty());

        // Test Display - should show value and modulus
        let display_str = alloc::format!("{}", a);
        assert!(display_str.contains("mod"));

        // Verify the retrieved value is correct
        assert_eq!(a.retrieve(), Uint::from(255_u64));
    }

    #[test]
    fn default_trait() {
        let default_val: F = Default::default();
        assert_eq!(default_val, F::ZERO);
        assert!(default_val.is_zero());
    }

    #[test]
    fn ord_trait() {
        let a: F = 10_u64.into();
        let b: F = 20_u64.into();
        let c: F = 10_u64.into();

        // Ord compares Montgomery form, not values
        // Just verify that equal values compare equal
        assert_eq!(a.cmp(&c), Ordering::Equal);
        assert_eq!(a.partial_cmp(&c), Some(Ordering::Equal));

        // Verify PartialOrd returns Some
        assert!(a.partial_cmp(&b).is_some());
    }

    #[allow(clippy::op_ref)]
    #[test]
    fn rem_operations() {
        let a: F = 17_u64.into();
        let b: F = 5_u64.into();

        // Test Rem
        let r = a % b;
        assert_eq!(r, F::from(2_u64));

        // Test Rem with reference
        let r2 = a % &b;
        assert_eq!(r2, F::from(2_u64));

        // Test RemAssign
        let mut c = a;
        c %= b;
        assert_eq!(c, F::from(2_u64));

        let mut d = a;
        d %= &b;
        assert_eq!(d, F::from(2_u64));
    }

    #[test]
    #[should_panic(expected = "Division by zero")]
    fn rem_by_zero_panics() {
        let a: F = 10_u64.into();
        let _ = a % F::zero();
    }

    #[test]
    fn shift_operations() {
        let a: F = 5_u64.into();

        // Test Shl
        let shifted_left = a << 2;
        assert_eq!(shifted_left, F::from(20_u64));

        // Test Shr
        let b: F = 20_u64.into();
        let shifted_right = b >> 2;
        assert_eq!(shifted_right, F::from(5_u64));
    }

    #[test]
    fn pow_operation() {
        let base: F = 2_u64.into();

        // Test basic exponentiation
        assert_eq!(base.pow(0), F::one());
        assert_eq!(base.pow(1), base);
        assert_eq!(base.pow(3), F::from(8_u64));
        assert_eq!(base.pow(10), F::from(1024_u64));

        // Test with different base
        let base3: F = 3_u64.into();
        assert_eq!(base3.pow(4), F::from(81_u64));
    }

    #[test]
    fn checked_operations() {
        let a: F = 10_u64.into();
        let b: F = 5_u64.into();

        // Test checked_neg
        assert_eq!(a.checked_neg().unwrap(), -a);

        // Test checked_add
        assert_eq!(a.checked_add(&b).unwrap(), a + b);

        // Test checked_sub
        assert_eq!(a.checked_sub(&b).unwrap(), a - b);

        // Test checked_mul
        assert_eq!(a.checked_mul(&b).unwrap(), a * b);

        // Test checked_div
        assert_eq!(a.checked_div(&b).unwrap(), a / b);
        assert!(a.checked_div(&F::zero()).is_none());

        // Test checked_rem
        assert_eq!(a.checked_rem(&b).unwrap(), a % b);
        assert!(a.checked_rem(&F::zero()).is_none());
    }

    #[test]
    fn div_assign_operation() {
        let mut a: F = 20_u64.into();
        let b: F = 4_u64.into();

        a /= b;
        assert_eq!(a * b, F::from(20_u64));

        let mut c: F = 30_u64.into();
        let d: F = 5_u64.into();
        c /= &d;
        assert_eq!(c * d, F::from(30_u64));
    }

    #[test]
    fn conversions() {
        // Test reference conversions from primitives
        let val_u32 = 42_u32;
        let a = F::from(&val_u32);
        assert_eq!(a, F::from(42_u64));

        let val_i64 = -100_i64;
        let b = F::from(&val_i64);
        assert_eq!(b, F::from(-100_i64));

        // Test Uint conversions
        let uint_val = Uint::<4>::from(123_u64);
        let c = F::from(&uint_val);
        assert_eq!(c, F::from(123_u64));

        // Test Int<N> conversions
        let int_val = Int::<4>::from(456_i64);
        let a = F::from(int_val);
        assert_eq!(a, F::from(456_u64));

        let int_ref = Int::<2>::from(-789_i64);
        let b = F::from(&int_ref);
        assert_eq!(b, F::from(-789_i64));

        // Test crypto_bigint::Int conversions
        let crypto_int = crypto_bigint::Int::<4>::from(321_i64);
        let c = F::from(crypto_int);
        assert_eq!(c, F::from(321_u64));

        let crypto_int_neg = crypto_bigint::Int::<2>::from(-654_i64);
        let d = F::from(&crypto_int_neg);
        assert_eq!(d, F::from(-654_i64));

        // Conversions from and to ConstMontyForm
        let mont_form = ConstMontyForm::new(&Uint::<{ U256::LIMBS }>::from(999_u64));
        let f: F = mont_form.into();
        let f2 = F::new(mont_form);
        assert_eq!(f, f2);
        assert_eq!(mont_form, f.into());
    }

    #[test]
    fn constant_time_selectable() {
        use crypto_bigint::subtle::Choice;

        let a: F = 10_u64.into();
        let b: F = 20_u64.into();

        // Test ConditionallySelectable
        let selected_a = F::conditional_select(&a, &b, Choice::from(0));
        assert_eq!(selected_a, a);

        let selected_b = F::conditional_select(&a, &b, Choice::from(1));
        assert_eq!(selected_b, b);
    }

    #[test]
    fn crypto_bigint_traits() {
        use crypto_bigint::{One as CryptoOne, Zero as CryptoZero};

        // Test crypto_bigint::Zero
        let zero: F = CryptoZero::zero();
        assert_eq!(zero, F::ZERO);

        // Test crypto_bigint::One
        let one: F = CryptoOne::one();
        assert_eq!(one, F::ONE);

        // Test Square
        let a: F = 7_u64.into();
        assert_eq!(a.square(), a * a);
    }

    #[test]
    fn retrieve_trait() {
        use crypto_bigint::modular::Retrieve;

        let a: F = 123_u64.into();
        let retrieved = Retrieve::retrieve(&a);
        assert_eq!(retrieved, Uint::from(123_u64));
    }

    #[test]
    fn constants() {
        assert_eq!(F::LIMBS, U256::LIMBS);
    }

    #[cfg(feature = "rand")]
    #[test]
    fn random_generation() {
        use rand::prelude::*;

        // Use a seeded RNG for reproducibility
        let mut rng = StdRng::seed_from_u64(1);

        // Test crypto_bigint::Random trait
        let random1: F = crypto_bigint::Random::random(&mut rng);
        let random2: F = crypto_bigint::Random::random(&mut rng);

        // Random values should be different
        assert_ne!(random1, random2);

        // Test Distribution trait
        let random3: F = rng.random();
        let random4: F = rng.random();

        assert_ne!(random3, random4);
    }
}

#[cfg(test)]
#[allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
mod prop_tests {
    use crypto_bigint::{U256, const_monty_params};
    use num_traits::{One, Zero};
    use proptest::prelude::*;

    use super::*;

    const_monty_params!(
        ModP,
        U256,
        "00dca94d8a1ecce3b6e8755d8999787d0524d8ca1ea755e7af84fb646fa31f27"
    );
    type F = ConstMontyField<ModP, { U256::LIMBS }>;

    fn any_f() -> impl Strategy<Value = F> {
        any::<u64>().prop_map(F::from)
    }

    fn any_nonzero_f() -> impl Strategy<Value = F> {
        any_f().prop_filter("non-zero", |x| !x.is_zero())
    }

    proptest! {
        #[test]
        fn prop_sum_over_concat_equals_sum_over_parts(a in proptest::collection::vec(any_f(), 0..20), b in proptest::collection::vec(any_f(), 0..20)) {
            let s_ab: F = a.iter().chain(b.iter()).cloned().sum();
            let s_a: F = a.iter().cloned().sum();
            let s_b: F = b.iter().cloned().sum();
            prop_assert_eq!(s_ab, s_a + s_b);
        }

        #[test]
        fn prop_product_over_concat_equals_product_over_parts(a in proptest::collection::vec(any_f(), 0..20), b in proptest::collection::vec(any_f(), 0..20)) {
            let p_ab: F = a.iter().chain(b.iter()).product();
            let p_a: F = a.iter().product();
            let p_b: F = b.iter().product();
            prop_assert_eq!(p_ab, p_a * p_b);
        }
        #[test]
        fn prop_add_commutative(a in any_f(), b in any_f()) {
            prop_assert_eq!(a + b, b + a);
        }

        #[test]
        fn prop_mul_commutative(a in any_f(), b in any_f()) {
            prop_assert_eq!(a * b, b * a);
        }

        #[test]
        fn prop_add_associative(a in any_f(), b in any_f(), c in any_f()) {
            prop_assert_eq!((a + b) + c, a + (b + c));
        }

        #[test]
        fn prop_mul_associative(a in any_f(), b in any_f(), c in any_f()) {
            prop_assert_eq!((a * b) * c, a * (b * c));
        }

        #[test]
        fn prop_distributive(a in any_f(), b in any_f(), c in any_f()) {
            prop_assert_eq!(a * (b + c), a * b + a * c);
            prop_assert_eq!((a + b) * c, a * c + b * c);
        }

        #[test]
        fn prop_additive_identity(a in any_f()) {
            prop_assert_eq!(a + F::zero(), a);
            prop_assert_eq!(F::zero() + a, a);
        }

        #[test]
        fn prop_multiplicative_identity(a in any_f()) {
            prop_assert_eq!(a * F::one(), a);
            prop_assert_eq!(F::one() * a, a);
        }

        #[test]
        fn prop_additive_inverse(a in any_f()) {
            prop_assert_eq!(a + (-a), F::zero());
        }

        #[test]
        fn prop_sub_roundtrip(a in any_f(), b in any_f()) {
            prop_assert_eq!(a - b + b, a);
        }

        #[test]
        fn prop_inversion_nonzero(a in any_nonzero_f()) {
            let inv = a.inv().expect("non-zero should invert");
            prop_assert_eq!(a * inv, F::one());
        }

        #[test]
        fn prop_division_matches_inverse(a in any_f(), b in any_nonzero_f()) {
            let inv_b = b.inv().unwrap();
            let lhs = a / b;
            let rhs = a * inv_b;
            prop_assert_eq!(lhs, rhs);
        }

        #[test]
        fn prop_assign_ops_equivalence(x0 in any_f(), y in any_f()) {
            let mut x = x0;
            x += y;
            prop_assert_eq!(x, x0 + y);
            let mut x2 = x0;
            x2 -= y;
            prop_assert_eq!(x2, x0 - y);
            let mut x3 = x0;
            x3 *= y;
            prop_assert_eq!(x3, x0 * y);
        }
    }
}
