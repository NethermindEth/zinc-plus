use super::*;
use crate::{
    IntRing, Semiring, boolean::Boolean, crypto_bigint_int::Int, crypto_bigint_uint::Uint,
};
use core::{
    cmp::Ordering,
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    hash::{Hash, Hasher},
    iter::{Product, Sum},
    ops::{Add, AddAssign, DivAssign, Mul, MulAssign, Rem, Sub, SubAssign},
};
use crypto_bigint::{
    NonZero, Odd, One,
    modular::{MontyForm, MontyParams},
};
use crypto_primitives_proc_macros::InfallibleCheckedOp;
use num_traits::{CheckedAdd, CheckedDiv, CheckedMul, CheckedNeg, CheckedSub, Pow};

#[derive(Clone, PartialEq, Eq, InfallibleCheckedOp)]
#[infallible_checked_unary_op((CheckedNeg, neg))]
#[infallible_checked_binary_op((CheckedAdd, add), (CheckedSub, sub), (CheckedMul, mul))]
#[repr(transparent)]
pub struct MontyField<const LIMBS: usize>(MontyForm<LIMBS>);

impl<const LIMBS: usize> MontyField<LIMBS> {
    /// Creates a new `MontyField` from a `MontyForm`.
    #[inline(always)]
    pub const fn new(form: MontyForm<LIMBS>) -> Self {
        Self(form)
    }

    #[inline(always)]
    pub const fn new_unchecked(inner: Uint<LIMBS>, config: &MontyParams<LIMBS>) -> Self {
        Self(MontyForm::from_montgomery(inner.into_inner(), *config))
    }

    #[inline(always)]
    pub const fn inner(&self) -> &Uint<LIMBS> {
        Uint::new_ref(self.0.as_montgomery())
    }

    #[inline(always)]
    pub const fn into_inner(self) -> Uint<LIMBS> {
        Uint::new(self.0.to_montgomery())
    }

    /// Retrieves the integer currently encoded in this [`MontyForm`],
    /// guaranteed to be reduced.
    pub const fn retrieve(&self) -> Uint<LIMBS> {
        Uint::new(self.0.retrieve())
    }

    /// Access the value in Montgomery form.
    pub const fn as_montgomery(&self) -> &Uint<LIMBS> {
        Uint::new_ref(self.0.as_montgomery())
    }

    /// Mutably access the value in Montgomery form.
    pub fn as_montgomery_mut(&mut self) -> &mut Uint<LIMBS> {
        Uint::new_ref_mut(self.0.as_montgomery_mut())
    }

    /// Create a `MontyField` from a value in Montgomery form.
    pub const fn from_montgomery(integer: Uint<LIMBS>, config: &MontyParams<LIMBS>) -> Self {
        Self(MontyForm::from_montgomery(integer.into_inner(), *config))
    }

    /// Extract the value from the `MontyForm` in Montgomery form.
    pub const fn to_montgomery(&self) -> Uint<LIMBS> {
        Uint::new(self.0.to_montgomery())
    }

    /// Performs division by 2, that is returns `x` such that `x + x = self`.
    pub const fn div_by_2(&self) -> Self {
        Self(self.0.div_by_2())
    }

    /// Double `self`.
    pub const fn double(&self) -> Self {
        Self(self.0.double())
    }

    /// See [MontyForm::pow_bounded_exp].
    pub const fn pow_bounded_exp<const RHS_LIMBS: usize>(
        &self,
        exponent: &Uint<RHS_LIMBS>,
        exponent_bits: u32,
    ) -> Self {
        Self(self.0.pow_bounded_exp(exponent.inner(), exponent_bits))
    }
}

//
// Core traits
//

impl<const LIMBS: usize> Debug for MontyField<LIMBS> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        Debug::fmt(&self.0, f)
    }
}

impl<const LIMBS: usize> Display for MontyField<LIMBS> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "{} (mod {})",
            self.0.retrieve(),
            self.0.params().modulus()
        )
    }
}

impl<const LIMBS: usize> PartialOrd for MontyField<LIMBS> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.modulus() != other.modulus() {
            return None;
        }
        Some(Ord::cmp(self.0.as_montgomery(), other.0.as_montgomery()))
    }
}

impl<const LIMBS: usize> Hash for MontyField<LIMBS> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.as_montgomery().hash(state)
    }
}

//
// Basic arithmetic operations
//

impl<const LIMBS: usize> Neg for MontyField<LIMBS> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        *self.0.as_montgomery_mut() = self
            .0
            .as_montgomery()
            .neg_mod(self.0.params().modulus().as_nz_ref());
        self
    }
}

macro_rules! impl_basic_op_forward_to_assign {
    ($trait:ident, $method:ident, $assign_method:ident) => {
        impl<const LIMBS: usize> $trait for MontyField<LIMBS> {
            type Output = MontyField<LIMBS>;

            #[inline(always)]
            fn $method(self, rhs: MontyField<LIMBS>) -> Self::Output {
                self.$method(&rhs)
            }
        }

        impl<const LIMBS: usize> $trait<&Self> for MontyField<LIMBS> {
            type Output = MontyField<LIMBS>;

            #[inline(always)]
            fn $method(mut self, rhs: &MontyField<LIMBS>) -> Self::Output {
                self.$assign_method(rhs);
                self
            }
        }

        impl<const LIMBS: usize> $trait for &MontyField<LIMBS> {
            type Output = MontyField<LIMBS>;

            #[inline(always)]
            fn $method(self, rhs: &MontyField<LIMBS>) -> Self::Output {
                self.clone().$method(rhs)
            }
        }

        impl<const LIMBS: usize> $trait<MontyField<LIMBS>> for &MontyField<LIMBS> {
            type Output = MontyField<LIMBS>;

            #[inline(always)]
            fn $method(self, rhs: MontyField<LIMBS>) -> Self::Output {
                self.clone().$method(&rhs)
            }
        }
    };
}

impl_basic_op_forward_to_assign!(Add, add, add_assign);
impl_basic_op_forward_to_assign!(Sub, sub, sub_assign);
impl_basic_op_forward_to_assign!(Mul, mul, mul_assign);
impl_basic_op_forward_to_assign!(Div, div, div_assign);

impl<const LIMBS: usize> Pow<u32> for MontyField<LIMBS> {
    type Output = Self;

    fn pow(self, rhs: u32) -> Self::Output {
        Self(self.0.pow(&crypto_bigint::Uint::<1>::from(rhs)))
    }
}

impl<const LIMBS: usize> Inv for MontyField<LIMBS> {
    type Output = Option<Self>;

    fn inv(self) -> Self::Output {
        Some(Self(Option::from(self.0.invert_vartime())?))
    }
}

impl<const LIMBS: usize> Inv for &MontyField<LIMBS> {
    type Output = Option<MontyField<LIMBS>>;

    fn inv(self) -> Self::Output {
        Some(MontyField(Option::from(self.0.invert_vartime())?))
    }
}

//
// Checked arithmetic operations
// (Note: Field operations do not overflow)
//

impl<const LIMBS: usize> CheckedDiv for MontyField<LIMBS> {
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn checked_div(&self, rhs: &Self) -> Option<Self> {
        Some(self * rhs.inv()?)
    }
}

//
// Arithmetic assign operations
//

macro_rules! impl_op_assign_boilerplate {
    ($trait:ident, $method:ident) => {
        impl<const LIMBS: usize> $trait for MontyField<LIMBS> {
            #[inline(always)]
            fn $method(&mut self, rhs: Self) {
                self.$method(&rhs);
            }
        }
    };
}

impl_op_assign_boilerplate!(AddAssign, add_assign);
impl_op_assign_boilerplate!(SubAssign, sub_assign);
impl_op_assign_boilerplate!(MulAssign, mul_assign);
impl_op_assign_boilerplate!(DivAssign, div_assign);

impl<const LIMBS: usize> AddAssign<&Self> for MontyField<LIMBS> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: &Self) {
        *self.0.as_montgomery_mut() = self
            .0
            .as_montgomery()
            .add_mod(rhs.0.as_montgomery(), self.0.params().modulus().as_nz_ref());
    }
}

impl<const LIMBS: usize> SubAssign<&Self> for MontyField<LIMBS> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &Self) {
        *self.0.as_montgomery_mut() = self
            .0
            .as_montgomery()
            .sub_mod(rhs.0.as_montgomery(), self.0.params().modulus().as_nz_ref());
    }
}

impl<const LIMBS: usize> MulAssign<&Self> for MontyField<LIMBS> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &Self) {
        let monty_mul = crypto_bigint_helpers::mul::monty_mul(
            self.0.as_montgomery(),
            rhs.0.as_montgomery(),
            self.0.params().modulus().as_ref(),
        );
        *self.0.as_montgomery_mut() = monty_mul;
    }
}

impl<const LIMBS: usize> DivAssign<&Self> for MontyField<LIMBS> {
    fn div_assign(&mut self, rhs: &Self) {
        self.mul_assign(rhs.inv().expect("Division by zero"));
    }
}

//
// Aggregate operations
//

impl<const LIMBS: usize> Sum for MontyField<LIMBS> {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let Some(MontyField(first)) = iter.next() else {
            panic!("Sum of an empty iterator is not defined for MontyField");
        };
        Self(iter.fold(first, |acc, x| MontyForm::add(&acc, &x.0)))
    }
}

impl<'a, const LIMBS: usize> Sum<&'a Self> for MontyField<LIMBS> {
    fn sum<I: Iterator<Item = &'a Self>>(mut iter: I) -> Self {
        let Some(MontyField(first)) = iter.next() else {
            panic!("Sum of an empty iterator is not defined for MontyField");
        };
        Self(iter.fold(*first, |acc, x| MontyForm::add(&acc, &x.0)))
    }
}

impl<const LIMBS: usize> Product for MontyField<LIMBS> {
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn product<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let Some(first) = iter.next() else {
            panic!("Product of an empty iterator is not defined for MontyField");
        };
        iter.fold(first, |acc, x| acc * x)
    }
}

impl<'a, const LIMBS: usize> Product<&'a Self> for MontyField<LIMBS> {
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn product<I: Iterator<Item = &'a Self>>(mut iter: I) -> Self {
        let Some(first) = iter.next() else {
            panic!("Product of an empty iterator is not defined for MontyField");
        };
        iter.fold(first.clone(), |acc, x| acc * x)
    }
}

//
// Conversions
//

impl<const LIMBS: usize> From<MontyForm<LIMBS>> for MontyField<LIMBS> {
    #[inline(always)]
    fn from(value: MontyForm<LIMBS>) -> Self {
        Self(value)
    }
}

impl<const LIMBS: usize> From<MontyField<LIMBS>> for MontyForm<LIMBS> {
    #[inline(always)]
    fn from(value: MontyField<LIMBS>) -> Self {
        value.0
    }
}

impl<const LIMBS: usize> From<&MontyField<LIMBS>> for MontyField<LIMBS> {
    fn from(value: &Self) -> Self {
        value.clone()
    }
}

macro_rules! impl_from_unsigned {
    ($($t:ty),* $(,)?) => {
        $(
            impl<const LIMBS: usize>FromWithConfig<$t> for MontyField<LIMBS> {
                fn from_with_cfg(value: $t, cfg: &Self::Config) -> Self {
                    let abs: crypto_bigint::Uint<LIMBS> = value.into();
                    let monty_mul = crypto_bigint_helpers::mul::monty_mul(
                        &abs,
                        cfg.r2(),
                        cfg.modulus(),
                    );
                    MontyField(MontyForm::from_montgomery(monty_mul, *cfg))
                }
            }

            impl<const LIMBS: usize>FromWithConfig<&$t> for MontyField<LIMBS> {
                fn from_with_cfg(value: &$t, cfg: &Self::Config) -> Self {
                    Self::from_with_cfg(*value, cfg)
                }
            }
        )*
    };
}

macro_rules! impl_from_signed {
    ($($t:ty),* $(,)?) => {
        $(
            #[allow(clippy::arithmetic_side_effects)] // False alert
            impl<const LIMBS: usize>FromWithConfig<$t> for MontyField<LIMBS> {
                fn from_with_cfg(value: $t, cfg: &Self::Config) -> Self {
                    let magnitude = Uint::from(value.abs_diff(0));
                    let monty_mul = crypto_bigint_helpers::mul::monty_mul(
                        magnitude.inner(),
                        cfg.r2(),
                        cfg.modulus(),
                    );
                    let result = MontyField(MontyForm::from_montgomery(monty_mul, *cfg));
                    if value.is_negative() { -result } else { result }
                }
            }

            impl<const LIMBS: usize>FromWithConfig<&$t> for MontyField<LIMBS> {
                fn from_with_cfg(value: &$t, cfg: &Self::Config) -> Self {
                    Self::from_with_cfg(*value, cfg)
                }
            }
        )*
    };
}

impl_from_unsigned!(u8, u16, u32, u64, u128);
impl_from_signed!(i8, i16, i32, i64, i128);

impl<const LIMBS: usize> FromWithConfig<bool> for MontyField<LIMBS> {
    fn from_with_cfg(value: bool, cfg: &Self::Config) -> Self {
        let value = if value {
            crypto_bigint::Uint::one()
        } else {
            Zero::zero()
        };
        Self(MontyForm::new(&value, *cfg))
    }
}

impl<const LIMBS: usize> FromWithConfig<&bool> for MontyField<LIMBS> {
    fn from_with_cfg(value: &bool, cfg: &Self::Config) -> Self {
        Self::from_with_cfg(*value, cfg)
    }
}

impl<const LIMBS: usize> FromWithConfig<Boolean> for MontyField<LIMBS> {
    fn from_with_cfg(value: Boolean, cfg: &Self::Config) -> Self {
        Self::from_with_cfg(*value, cfg)
    }
}

impl<const LIMBS: usize> FromWithConfig<&Boolean> for MontyField<LIMBS> {
    fn from_with_cfg(value: &Boolean, cfg: &Self::Config) -> Self {
        Self::from_with_cfg(*value, cfg)
    }
}

impl<const LIMBS: usize, const LIMBS2: usize> FromWithConfig<Int<LIMBS2>> for MontyField<LIMBS> {
    fn from_with_cfg(value: Int<LIMBS2>, cfg: &Self::Config) -> Self {
        Self::from_with_cfg(&value, cfg)
    }
}

impl<const LIMBS: usize, const LIMBS2: usize> FromWithConfig<&Int<LIMBS2>> for MontyField<LIMBS> {
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn from_with_cfg(value: &Int<LIMBS2>, cfg: &Self::Config) -> Self {
        let mut abs = value.inner().abs();
        if LIMBS < LIMBS2 {
            abs = abs.rem(cfg.modulus().get().resize::<LIMBS2>())
        };
        let abs = abs.resize();

        let monty_mul = crypto_bigint_helpers::mul::monty_mul(&abs, cfg.r2(), cfg.modulus());
        let result = MontyField(MontyForm::from_montgomery(monty_mul, *cfg));

        if value.is_negative() { -result } else { result }
    }
}

impl<const LIMBS: usize, const LIMBS2: usize> FromWithConfig<Uint<LIMBS2>> for MontyField<LIMBS> {
    fn from_with_cfg(value: Uint<LIMBS2>, cfg: &Self::Config) -> Self {
        Self::from_with_cfg(&value, cfg)
    }
}

impl<const LIMBS: usize, const LIMBS2: usize> FromWithConfig<&Uint<LIMBS2>> for MontyField<LIMBS> {
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn from_with_cfg(value: &Uint<LIMBS2>, cfg: &Self::Config) -> Self {
        let value: crypto_bigint::Uint<LIMBS> = if LIMBS >= LIMBS2 {
            value.inner().resize()
        } else {
            value
                .inner()
                .rem(&NonZero::<crypto_bigint::Uint<LIMBS>>::new_unwrap(
                    cfg.modulus().get(),
                ))
                .resize()
        };

        let monty_mul = crypto_bigint_helpers::mul::monty_mul(&value, cfg.r2(), cfg.modulus());
        MontyField(MontyForm::from_montgomery(monty_mul, *cfg))
    }
}

//
// Semiring, Ring and Field
//

impl<const LIMBS: usize> Semiring for MontyField<LIMBS> {}

impl<const LIMBS: usize> Ring for MontyField<LIMBS> {}

impl<const LIMBS: usize> Field for MontyField<LIMBS> {
    type Inner = Uint<LIMBS>;
    type Modulus = Self::Inner;

    #[inline(always)]
    fn inner(&self) -> &Self::Inner {
        Uint::new_ref(self.0.as_montgomery())
    }

    #[inline(always)]
    fn inner_mut(&mut self) -> &mut Self::Inner {
        Uint::new_ref_mut(self.0.as_montgomery_mut())
    }

    #[inline(always)]
    fn into_inner(self) -> Self::Inner {
        Uint::new(self.0.to_montgomery())
    }
}

impl<const LIMBS: usize> PrimeField for MontyField<LIMBS> {
    type Config = MontyParams<LIMBS>;

    fn cfg(&self) -> &Self::Config {
        self.0.params()
    }

    fn is_zero(value: &Self) -> bool {
        value.0.as_montgomery().is_zero()
    }

    fn modulus(&self) -> Self::Modulus {
        Uint::new(self.0.params().modulus().get())
    }

    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn modulus_minus_one_div_two(&self) -> Self::Inner {
        let value = self.0.params().modulus().get();
        Uint::new(
            (value - crypto_bigint::Uint::one())
                / NonZero::new(crypto_bigint::Uint::<LIMBS>::from(2_u8)).unwrap(),
        )
    }

    fn make_cfg(modulus: &Self::Modulus) -> Result<Self::Config, FieldError> {
        let Some(modulus) = Odd::new(*modulus.inner()).into_option() else {
            return Err(FieldError::InvalidModulus);
        };
        Ok(MontyParams::new(modulus))
    }

    fn new_with_cfg(inner: Self::Inner, cfg: &Self::Config) -> Self {
        Self(MontyForm::new(inner.inner(), *cfg))
    }

    fn new_unchecked_with_cfg(inner: Self::Inner, cfg: &Self::Config) -> Self {
        Self(MontyForm::from_montgomery(inner.into_inner(), *cfg))
    }

    fn zero_with_cfg(cfg: &Self::Config) -> Self {
        Self(MontyForm::zero(*cfg))
    }

    fn one_with_cfg(cfg: &Self::Config) -> Self {
        Self(MontyForm::one(*cfg))
    }
}

//
// Zeroize
//

#[cfg(feature = "zeroize")]
impl<const LIMBS: usize> zeroize::Zeroize for MontyField<LIMBS> {
    fn zeroize(&mut self) {
        self.0.zeroize()
    }
}

//
// Predefined fields of various sizes for convenience
//

pub type F64 = MontyField<{ crypto_bigint::U64::LIMBS }>;
pub type F128 = MontyField<{ 2 * WORD_FACTOR }>;
pub type F192 = MontyField<{ 3 * WORD_FACTOR }>;
pub type F256 = MontyField<{ 4 * WORD_FACTOR }>;
pub type F320 = MontyField<{ 5 * WORD_FACTOR }>;
pub type F384 = MontyField<{ 6 * WORD_FACTOR }>;
pub type F448 = MontyField<{ 7 * WORD_FACTOR }>;
pub type F512 = MontyField<{ 8 * WORD_FACTOR }>;
pub type F576 = MontyField<{ 9 * WORD_FACTOR }>;
pub type F640 = MontyField<{ 10 * WORD_FACTOR }>;
pub type F704 = MontyField<{ 11 * WORD_FACTOR }>;
pub type F768 = MontyField<{ 12 * WORD_FACTOR }>;
pub type F832 = MontyField<{ 13 * WORD_FACTOR }>;
pub type F896 = MontyField<{ 14 * WORD_FACTOR }>;
pub type F960 = MontyField<{ 15 * WORD_FACTOR }>;
pub type F1024 = MontyField<{ 16 * WORD_FACTOR }>;
pub type F1280 = MontyField<{ 20 * WORD_FACTOR }>;
pub type F1536 = MontyField<{ 24 * WORD_FACTOR }>;
pub type F1792 = MontyField<{ 28 * WORD_FACTOR }>;
pub type F2048 = MontyField<{ 32 * WORD_FACTOR }>;
pub type F3072 = MontyField<{ 48 * WORD_FACTOR }>;
pub type F4096 = MontyField<{ 64 * WORD_FACTOR }>;
pub type F6144 = MontyField<{ 96 * WORD_FACTOR }>;
pub type F8192 = MontyField<{ 128 * WORD_FACTOR }>;
pub type F16384 = MontyField<{ 256 * WORD_FACTOR }>;
pub type F32768 = MontyField<{ 512 * WORD_FACTOR }>;

#[allow(clippy::arithmetic_side_effects, clippy::cast_lossless)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ensure_type_implements_trait;
    use alloc::vec;
    use crypto_bigint::U64;
    use num_traits::{ConstOne, ConstZero, Pow};

    const LIMBS: usize = 4;
    type F = F256;

    //
    // Test helpers
    //
    fn test_config() -> MontyParams<LIMBS> {
        // Using a 256-bit prime 2^256 - 2^32 - 977 (secp256k1 field prime)
        let modulus = crypto_bigint::Uint::<LIMBS>::from_be_hex(
            "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f",
        );
        let modulus = Odd::new(modulus).expect("modulus should be odd");
        MontyParams::new(modulus)
    }

    #[test]
    fn new_with_cfg_correct() {
        let x =
            Uint::from_be_hex("fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc30");

        let y = F::new_with_cfg(x, &test_config());

        assert_eq!(y, F::one_with_cfg(&test_config()));
    }

    fn from_u64(value: u64) -> F {
        F::from_with_cfg(value, &test_config())
    }

    fn from_i64(value: i64) -> F {
        F::from_with_cfg(value, &test_config())
    }

    fn zero() -> F {
        F::zero_with_cfg(&test_config())
    }

    fn one() -> F {
        F::one_with_cfg(&test_config())
    }

    #[test]
    fn ensure_blanket_traits() {
        // NB: this ensures `PrimeField` implementation too!
        ensure_type_implements_trait!(F, FromPrimitiveWithConfig);
    }

    #[test]
    fn zero_one_basics() {
        let z = zero();
        assert!(F::is_zero(&z));
        let o = one();
        assert!(!F::is_zero(&o));
        assert_ne!(z, o);
    }

    #[test]
    fn basic_operations() {
        // Negation
        let a = from_u64(9);
        let neg_a = -a.clone();
        assert_eq!(&a + &neg_a, zero());

        let a = from_u64(10);
        let b = from_u64(5);

        // Addition
        let c = a.clone() + b.clone();
        assert_eq!(c, from_u64(15));

        // Subtraction
        let d = a.clone() - b.clone();
        assert_eq!(d, from_u64(5));

        // Multiplication
        let e = a.clone() * b.clone();
        assert_eq!(e, from_u64(50));

        // Division
        let num = from_u64(11);
        let den = from_u64(5);
        let q = num.clone() / den.clone();
        assert_eq!(&q * &den, num);
    }

    #[test]
    fn basic_operations_overflow() {
        let params = test_config();
        let mod_minus_one = Uint::new(params.modulus().get() - crypto_bigint::Uint::one());
        let mod_minus_one = F::from_with_cfg(mod_minus_one, &params);

        // Negation
        let res = -mod_minus_one.clone();
        assert_eq!(res, one());

        // Addition
        let res = mod_minus_one.clone() + one();
        assert_eq!(res, zero());

        // Subtraction
        let res = zero() - one();
        assert_eq!(res, mod_minus_one);

        // Multiplication
        let res = mod_minus_one.clone() * from_u64(2);
        assert_eq!(res, mod_minus_one.clone() - one());

        let res = mod_minus_one.clone() * mod_minus_one.clone();
        assert_eq!(res, one());

        // Division
        let res = one() / mod_minus_one.clone();
        assert_eq!(res, mod_minus_one);
    }

    #[test]
    fn add_wrapping() {
        let a = from_i64(-100);
        let b = from_u64(105);
        let c = a + b;
        let d = from_u64(5);
        assert_eq!(c, d);
    }

    #[allow(clippy::op_ref)]
    #[test]
    fn reference_operations() {
        let a = from_u64(10);
        let b = from_u64(5);

        // Addition
        let c = &a + &b;
        assert_eq!(c, from_u64(15));

        // Subtraction
        let d = &a - &b;
        assert_eq!(d, from_u64(5));

        // Multiplication
        let e = &a * &b;
        assert_eq!(e, from_u64(50));

        // Division
        let num = from_u64(11);
        let den = from_u64(5);
        let q = &num / &den;
        assert_eq!(&q * &den, num);
    }

    #[test]
    fn from_bool() {
        let cfg = test_config();
        assert_eq!(F::from_with_cfg(true, &cfg), one());
        assert_eq!(F::from_with_cfg(false, &cfg), zero());

        let t = F::from_with_cfg(true, &cfg);
        let f = F::from_with_cfg(false, &cfg);
        assert_eq!(t, one());
        assert_eq!(f, zero());
    }

    #[test]
    fn from_unsigned_and_signed() {
        const LIMBS: usize = U64::LIMBS;
        type F = F64;
        let cfg = F::make_cfg(&Uint::from(10064419296686275259_u64)).unwrap();
        macro_rules! to_field {
            ($x:expr) => {
                F::from_with_cfg($x, &cfg)
            };
        }
        let zero = F::zero_with_cfg(&cfg);
        let one = F::one_with_cfg(&cfg);
        assert_eq!(to_field!(0), zero);
        assert_eq!(to_field!(1), one);
        assert_eq!(to_field!(-1) + one, zero);
        assert_eq!(to_field!(-5) + F::from_with_cfg(5, &cfg), zero);

        // u64 maximum value (hand-calculated)
        assert_eq!(
            to_field!(u64::MAX),
            to_field!(Uint::<LIMBS>::from_be_hex("7453fff9266d8544"))
        );

        // i64 maximum value (hand-calculated)
        assert_eq!(
            to_field!(i64::MAX),
            to_field!(Uint::<LIMBS>::from_be_hex("7fffffffffffffff"))
        );

        // i64 minimum value (hand-calculated)
        assert_eq!(
            to_field!(i64::MIN),
            to_field!(Uint::<LIMBS>::from_be_hex("0bac0006d9927abb"))
        );

        // Verify property: i64::MIN + |i64::MIN| = 0
        let i64_min_abs = to_field!(i64::MIN.unsigned_abs());
        assert_eq!(to_field!(i64::MIN) + i64_min_abs, zero);
    }

    #[test]
    fn from_uint_and_int() {
        const LIMBS: usize = U64::LIMBS;
        type F = F64;
        let cfg = F::make_cfg(&Uint::from(10064419296686275259_u64)).unwrap();
        macro_rules! to_field {
            ($x:expr) => {
                F::from_with_cfg($x, &cfg)
            };
        }

        assert_eq!(
            Int::<LIMBS>::MIN.into_inner().abs(),
            Uint::<LIMBS>::MAX.into_inner() / Uint::<LIMBS>::from_u64(2).into_inner()
                + Uint::ONE.into_inner()
        );

        let u: Uint<LIMBS> = Uint::from(123_u64);
        assert_eq!(to_field!(u), to_field!(123_u64));

        let i: Int<LIMBS> = Int::from(123_i64);
        assert_eq!(to_field!(i), to_field!(123_u64));

        assert_eq!(to_field!(Uint::<LIMBS>::ZERO), F::zero_with_cfg(&cfg));

        // Uint maximum value (hand-calculated)
        assert_eq!(
            to_field!(u64::MAX),
            to_field!(Uint::<LIMBS>::from_be_hex("7453fff9266d8544"))
        );

        // Int maximum value (hand-calculated)
        assert_eq!(
            to_field!(i64::MAX),
            to_field!(Uint::<LIMBS>::from_be_hex("7fffffffffffffff"))
        );

        // Int minimum value (hand-calculated)
        assert_eq!(
            to_field!(i64::MIN),
            to_field!(Uint::<LIMBS>::from_be_hex("0bac0006d9927abb"))
        );
    }

    #[test]
    fn assign_operations() {
        // Addition
        let mut a = from_u64(5);
        a += from_u64(6);
        assert_eq!(a, from_u64(11));

        // Subtraction
        let mut a = from_u64(20);
        a -= from_u64(7);
        assert_eq!(a, from_u64(13));

        // Multiplication
        let mut a = from_u64(11);
        a *= from_u64(3);
        assert_eq!(a, from_u64(33));

        // Division
        let mut a = from_u64(20);
        let b = from_u64(4);
        a /= b;
        assert_eq!(&a * &from_u64(4), from_u64(20));
    }

    #[test]
    #[should_panic(expected = "Division by zero")]
    fn div_by_zero_panics() {
        let a = from_u64(7);
        let zero = zero();
        let _ = a / zero;
    }

    #[test]
    fn pow_operation() {
        // Test basic exponentiation
        let base = from_u64(2);

        // 2^0 = 1
        assert_eq!(base.clone().pow(0), one());

        // 2^1 = 2
        assert_eq!(base.clone().pow(1), base);

        // 2^3 = 8
        assert_eq!(base.clone().pow(3), from_u64(8));

        // 2^10 = 1024
        assert_eq!(base.clone().pow(10), from_u64(1024));

        // Test with different base
        let base = from_u64(3);

        // 3^4 = 81
        assert_eq!(base.clone().pow(4), from_u64(81));

        // Test with base 1
        let base = one();
        assert_eq!(base.clone().pow(1000), one());

        // Test with base 0
        let base = zero();
        assert_eq!(base.clone().pow(0), one()); // 0^0 = 1 by convention
        assert_eq!(base.clone().pow(10), zero()); // 0^n = 0 for n > 0
    }

    #[test]
    fn inv_operation() {
        let a = from_u64(5);
        let inv_a = a.clone().inv().unwrap();
        assert_eq!(&a * &inv_a, one());

        // Test that zero has no inverse
        let zero = zero();
        assert!(zero.inv().is_none());
    }

    #[test]
    fn checked_neg() {
        // Test with positive number
        let a = from_u64(10);
        let neg_a = a.checked_neg().unwrap();
        assert_eq!(neg_a, from_i64(-10));

        // Test with negative number
        let b = from_i64(-5);
        let neg_b = b.checked_neg().unwrap();
        assert_eq!(neg_b, from_u64(5));

        // Test with zero
        let zero_val = zero();
        let neg_zero = zero_val.checked_neg().unwrap();
        assert_eq!(neg_zero, zero());
    }

    #[test]
    fn checked_add() {
        let a = from_u64(10);
        let b = from_u64(5);

        let c = a.checked_add(&b).unwrap();
        assert_eq!(c, from_u64(15));
    }

    #[test]
    fn checked_sub() {
        let a = from_u64(10);
        let b = from_u64(5);

        let d = a.checked_sub(&b).unwrap();
        assert_eq!(d, from_u64(5));
    }

    #[test]
    fn checked_mul() {
        let a = from_u64(10);
        let b = from_u64(5);

        let e = a.checked_mul(&b).unwrap();
        assert_eq!(e, from_u64(50));
    }

    #[test]
    fn checked_div() {
        let a = from_u64(10);
        let b = from_u64(5);
        let zero = zero();

        // Normal division
        let c = a.checked_div(&b).unwrap();
        assert_eq!(&c * &b, a);

        // Division by zero
        assert!(a.checked_div(&zero).is_none());
    }

    #[allow(clippy::op_ref)]
    #[test]
    fn ref_and_value_combinations_add_sub_mul() {
        let a = from_u64(42);
        let b = from_u64(123);

        let r1 = &a + &b;
        let a1 = from_u64(42);
        let b1 = from_u64(123);
        let r2 = a1 + b1;
        let a2 = from_u64(42);
        let b2 = from_u64(123);
        let r3 = a2 + &b2;
        let a3 = from_u64(42);
        let b3 = from_u64(123);
        let r4 = &a3 + b3;
        assert_eq!(r1, r2);
        assert_eq!(r1, r3);
        assert_eq!(r1, r4);

        let a = from_u64(88);
        let b = from_u64(59);
        let s1 = &a - &b;
        let a1 = from_u64(88);
        let b1 = from_u64(59);
        let s2 = a1 - b1;
        let a2 = from_u64(88);
        let b2 = from_u64(59);
        let s3 = a2 - &b2;
        let a3 = from_u64(88);
        let b3 = from_u64(59);
        let s4 = &a3 - b3;
        assert_eq!(s1, s2);
        assert_eq!(s1, s3);
        assert_eq!(s1, s4);

        let a = from_u64(9);
        let b = from_u64(14);
        let m1 = &a * &b;
        let a1 = from_u64(9);
        let b1 = from_u64(14);
        let m2 = a1 * b1;
        let a2 = from_u64(9);
        let b2 = from_u64(14);
        let m3 = a2 * &b2;
        let a3 = from_u64(9);
        let b3 = from_u64(14);
        let m4 = &a3 * b3;
        assert_eq!(m1, m2);
        assert_eq!(m1, m3);
        assert_eq!(m1, m4);
    }

    #[test]
    fn assign_ops_with_refs_and_val() {
        let mut a = from_u64(100);
        let b = from_u64(50);
        a += b;
        assert_eq!(a, from_u64(150));

        let mut c = from_u64(100);
        let d = from_u64(50);
        c += &d;
        assert_eq!(c, from_u64(150));

        let mut e = from_u64(100);
        let f = from_u64(30);
        e -= f;
        assert_eq!(e, from_u64(70));

        let mut g = from_u64(100);
        let h = from_u64(30);
        g -= &h;
        assert_eq!(g, from_u64(70));

        let mut i = from_u64(10);
        let j = from_u64(5);
        i *= j;
        assert_eq!(i, from_u64(50));

        let mut k = from_u64(10);
        let l = from_u64(5);
        k *= &l;
        assert_eq!(k, from_u64(50));
    }

    #[test]
    fn aggregate_operations() {
        // Sum
        let values = vec![from_u64(1), from_u64(2), from_u64(3)];
        let sum: F = values.iter().sum();
        assert_eq!(sum, from_u64(6));

        let sum2: F = values.into_iter().sum();
        assert_eq!(sum2, from_u64(6));

        // Product
        let values = vec![from_u64(2), from_u64(3), from_u64(4)];
        let product: F = values.iter().product();
        assert_eq!(product, from_u64(24));

        let product2: F = values.into_iter().product();
        assert_eq!(product2, from_u64(24));

        // Test empty collections panic
        // Note: MontyField doesn't have a default config, so empty
        // iterators must panic
    }

    #[test]
    fn conversions() {
        let cfg = test_config();

        // Test FromWithConfig for BoxedUint
        let f = F::from_with_cfg(123_u64, &cfg);
        assert_eq!(f, from_u64(123));

        // Test inner() and cfg()
        let value = from_u64(456);
        let _inner_ref = value.inner();
        let _cfg_ref = value.cfg();
    }

    #[test]
    fn from_primitive() {
        let cfg = test_config();

        // Test FromWithConfig for various types
        let a = F::from_with_cfg(42_u8, &cfg);
        assert_eq!(a, from_u64(42));

        let b = F::from_with_cfg(12345_u16, &cfg);
        assert_eq!(b, from_u64(12345));

        let c = F::from_with_cfg(1234567890_u32, &cfg);
        assert_eq!(c, from_u64(1234567890));

        let d = F::from_with_cfg(1234567890123456789_u64, &cfg);
        assert_eq!(d, from_u64(1234567890123456789));

        let e = F::from_with_cfg(-42_i8, &cfg);
        assert_eq!(e, from_i64(-42));

        let f = F::from_with_cfg(-12345_i16, &cfg);
        assert_eq!(f, from_i64(-12345));

        let g = F::from_with_cfg(-1234567_i32, &cfg);
        assert_eq!(g, from_i64(-1234567));

        let h = F::from_with_cfg(-1234567890123456789_i64, &cfg);
        assert_eq!(h, from_i64(-1234567890123456789));
    }

    #[test]
    fn clone_works() {
        let a = from_u64(42);
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn equality_and_ordering() {
        let a = from_u64(10);
        let b = from_u64(10);
        let c = from_u64(20);

        // Test equality
        assert_eq!(a, b);
        assert_ne!(a, c);

        // Test ordering
        // Note: ordering is based on Montgomery representation, not value
        // We just test consistency of ordering
        assert_eq!(a.partial_cmp(&b), Some(Ordering::Equal));
        assert_ne!(a.partial_cmp(&c), Some(Ordering::Equal));
        assert!(a <= b);
        assert!(a >= b);

        // Verify transitivity of ordering
        let d = from_u64(30);
        if a < c && c < d {
            assert!(a < d);
        }
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

        let a = from_u64(42);
        let b = from_u64(42);

        let mut hasher_a = TestHasher { state: 0 };
        a.hash(&mut hasher_a);
        let hash_a = hasher_a.finish();

        let mut hasher_b = TestHasher { state: 0 };
        b.hash(&mut hasher_b);
        let hash_b = hasher_b.finish();

        // Equal values should have equal hashes
        assert_eq!(hash_a, hash_b);
    }

    // MontyField-specific tests

    #[test]
    fn prime_field_methods() {
        let a = from_u64(42);
        let cfg = a.cfg();

        // Test that we can get modulus
        let modulus = a.modulus();
        assert_eq!(
            modulus,
            Uint::new(crypto_bigint::Uint::<LIMBS>::from_be_hex(
                "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f"
            ))
        );

        // Test modulus_minus_one_div_two
        let m_minus_1_div_2 = a.modulus_minus_one_div_two();
        assert_eq!(
            m_minus_1_div_2,
            Uint::new(
                (crypto_bigint::Uint::<LIMBS>::from_be_hex(
                    "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f"
                ) - crypto_bigint::Uint::one())
                    / crypto_bigint::Uint::<LIMBS>::from(2u64)
            )
        );

        // Test zero_with_cfg and one_with_cfg
        let z = F::zero_with_cfg(cfg);
        assert!(F::is_zero(&z));
        let o = F::one_with_cfg(cfg);
        assert!(!F::is_zero(&o));
    }

    #[test]
    fn make_cfg_works() {
        let modulus = Uint::new(crypto_bigint::Uint::<LIMBS>::from_be_hex(
            "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f",
        ));
        let cfg = F::make_cfg(&modulus).expect("Should create config");

        // Create a field element using this config
        let a = F::from_with_cfg(123_u64, &cfg);
        assert_eq!(a, from_u64(123));
    }

    #[test]
    fn make_cfg_rejects_even_modulus() {
        let even_modulus = Uint::<LIMBS>::from(42_u64);
        let result = F::make_cfg(&even_modulus);
        assert!(result.is_err());
    }
}
