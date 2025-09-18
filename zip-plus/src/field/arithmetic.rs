#![allow(clippy::arithmetic_side_effects)]

use crypto_bigint::{
    Uint,
    modular::{ConstMontyForm, ConstMontyParams},
};
use num_traits::{
    CheckedAdd, CheckedDiv, CheckedMul, CheckedNeg, CheckedRem, CheckedSub, ConstOne, ConstZero,
    Inv, Pow, Zero,
};
use std::{
    iter::{Product, Sum},
    ops::{
        Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Shl, Shr, Sub,
        SubAssign,
    },
};

use crate::{field::ConstMontyField, rem, utils::WORD_FACTOR};

// Macro to implement arithmetic traits for Field in all ref/value combinations
// Uses the same method name for both the trait method and the inner explicit
// call.
macro_rules! impl_field_op {
    ($trait:ident, $method:ident) => {
        impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> $trait
            for ConstMontyField<Mod, LIMBS>
        {
            type Output = Self;
            fn $method(self, rhs: Self) -> Self::Output {
                Self(self.0.$method(rhs.0))
            }
        }
        impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> $trait<&Self>
            for ConstMontyField<Mod, LIMBS>
        {
            type Output = Self;
            fn $method(self, rhs: &Self) -> Self::Output {
                Self(self.0.$method(&rhs.0))
            }
        }
        impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> $trait
            for &ConstMontyField<Mod, LIMBS>
        {
            type Output = ConstMontyField<Mod, LIMBS>;
            fn $method(self, rhs: Self) -> Self::Output {
                ConstMontyField(self.0.$method(rhs.0))
            }
        }
        impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> $trait<ConstMontyField<Mod, LIMBS>>
            for &ConstMontyField<Mod, LIMBS>
        {
            type Output = ConstMontyField<Mod, LIMBS>;
            fn $method(self, rhs: ConstMontyField<Mod, LIMBS>) -> Self::Output {
                ConstMontyField(self.0.$method(&rhs.0))
            }
        }
    };
}

impl_field_op!(Add, add);
impl_field_op!(Sub, sub);
impl_field_op!(Mul, mul);

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> Neg for ConstMontyField<Mod, LIMBS> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(self.0.neg())
    }
}

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> Inv for ConstMontyField<Mod, LIMBS> {
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

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> Div for ConstMontyField<Mod, LIMBS> {
    type Output = Self;

    #[allow(clippy::op_ref)]
    fn div(self, rhs: Self) -> Self::Output {
        self / &rhs
    }
}

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> Div<&Self> for ConstMontyField<Mod, LIMBS> {
    type Output = Self;

    fn div(self, rhs: &Self) -> Self::Output {
        self.checked_div(rhs).expect("Division by zero")
    }
}

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> Rem for ConstMontyField<Mod, LIMBS> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        Self(ConstMontyForm::new(&rem!(
            self.0.retrieve(),
            &rhs.0.retrieve()
        )))
    }
}

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> Rem<&Self> for ConstMontyField<Mod, LIMBS> {
    type Output = Self;

    fn rem(self, rhs: &Self) -> Self::Output {
        self % *rhs
    }
}

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> Pow<u32> for ConstMontyField<Mod, LIMBS> {
    type Output = Self;

    fn pow(self, rhs: u32) -> Self::Output {
        Self(self.0.pow(&Uint::<WORD_FACTOR>::from(rhs)))
    }
}

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> Shl<u32> for ConstMontyField<Mod, LIMBS> {
    type Output = Self;

    fn shl(mut self, rhs: u32) -> Self::Output {
        let mut value = self.0.retrieve();
        value <<= rhs;
        self.0 = ConstMontyForm::new(&value);
        self
    }
}

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> Shr<u32> for ConstMontyField<Mod, LIMBS> {
    type Output = Self;

    fn shr(mut self, rhs: u32) -> Self::Output {
        let mut value = self.0.retrieve();
        value >>= rhs;
        self.0 = ConstMontyForm::new(&value);
        self
    }
}

//
// Checked arithmetic traits
//

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> CheckedNeg for ConstMontyField<Mod, LIMBS> {
    fn checked_neg(&self) -> Option<Self> {
        Some(Self(-self.0))
    }
}

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> CheckedAdd for ConstMontyField<Mod, LIMBS> {
    fn checked_add(&self, rhs: &Self) -> Option<Self> {
        Some(*self + rhs)
    }
}

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> CheckedSub for ConstMontyField<Mod, LIMBS> {
    fn checked_sub(&self, rhs: &Self) -> Option<Self> {
        Some(*self - rhs)
    }
}

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> CheckedMul for ConstMontyField<Mod, LIMBS> {
    fn checked_mul(&self, rhs: &Self) -> Option<Self> {
        Some(*self * rhs)
    }
}

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> CheckedDiv for ConstMontyField<Mod, LIMBS> {
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

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> CheckedRem for ConstMontyField<Mod, LIMBS> {
    fn checked_rem(&self, v: &Self) -> Option<Self> {
        if v.is_zero() { None } else { Some(*self % v) }
    }
}

//
// Assignment arithmetic traits
//

// Macro to implement assignment arithmetic traits (e.g., AddAssign) for Field
// Implements both rhs: Self and rhs: &Self, using an explicit inner method on
// ConstConstMontyParamsForm
macro_rules! impl_field_op_assign {
    ($trait:ident, $method:ident, $inner:ident) => {
        impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> $trait
            for ConstMontyField<Mod, LIMBS>
        {
            fn $method(&mut self, rhs: Self) {
                // Use reference for inner call to avoid moves of rhs.0 where not needed
                *self = self.$inner(&rhs);
            }
        }
        impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> $trait<&Self>
            for ConstMontyField<Mod, LIMBS>
        {
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
// Aggregate arithmetic traits
//

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> Sum for ConstMontyField<Mod, LIMBS> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> Product for ConstMontyField<Mod, LIMBS> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}

impl<'a, Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> Sum<&'a Self>
    for ConstMontyField<Mod, LIMBS>
{
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl<'a, Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> Product<&'a Self>
    for ConstMontyField<Mod, LIMBS>
{
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}

#[cfg(test)]
mod tests {
    use crypto_bigint::{U256, const_monty_params};
    use num_traits::{One, Zero};

    use super::*;

    const_monty_params!(
        ModP,
        U256,
        "00dca94d8a1ecce3b6e8755d8999787d0524d8ca1ea755e7af84fb646fa31f27"
    );
    type F = ConstMontyField<ModP, { U256::LIMBS }>;

    #[test]
    fn add_wrapping_and_basic() {
        let a: F = (-100i64).into();
        let b: F = 105u64.into();
        let c = a + b;
        let d: F = 5u64.into();
        assert_eq!(c, d);
    }

    #[test]
    fn sub_basic() {
        let a: F = 100u64.into();
        let b: F = 7u64.into();
        assert_eq!(a - b, 93u64.into());
    }

    #[test]
    fn mul_basic() {
        let a: F = 100u64.into();
        let b: F = 7u64.into();
        assert_eq!(a * b, 700u64.into());
    }

    #[test]
    fn add_assign_basic() {
        let mut a: F = 5u64.into();
        a += F::from(6u64);
        assert_eq!(a, 11u64.into());
    }

    #[test]
    fn mul_assign_basic() {
        let mut a: F = 11u64.into();
        a *= F::from(3u64);
        assert_eq!(a, 33u64.into());
    }

    #[test]
    fn neg_basic() {
        let a: F = 9u64.into();
        let neg_a = -a;

        assert_eq!(a + neg_a, F::zero());
    }

    #[test]
    fn div_basic() {
        let num: F = 11u64.into();
        let den: F = 5u64.into();
        let q = num / den;
        assert_eq!(q * den, num);
    }

    #[test]
    #[should_panic]
    fn div_by_zero_returns_panics() {
        let a: F = 7u64.into();
        let zero = F::zero();
        let _ = a / zero;
    }

    #[test]
    fn ref_and_value_combinations_add_sub_mul() {
        let a: F = 42u64.into();
        let b: F = 123u64.into();

        let r1 = a + b;
        let a1: F = 42u64.into();
        let b1: F = 123u64.into();
        let r2 = a1 + b1;
        let r3 = a1 + b1;
        let a2: F = 42u64.into();
        let b2: F = 123u64.into();
        let r4 = a2 + b2;
        assert_eq!(r1, r2);
        assert_eq!(r1, r3);
        assert_eq!(r1, r4);

        let a: F = 88u64.into();
        let b: F = 59u64.into();
        let s1 = a - b;
        let a1: F = 88u64.into();
        let b1: F = 59u64.into();
        let s2 = a1 - b1;
        let s3 = a1 - b1;
        let a2: F = 88u64.into();
        let b2: F = 59u64.into();
        let s4 = a2 - b2;
        assert_eq!(s1, s2);
        assert_eq!(s1, s3);
        assert_eq!(s1, s4);

        let a: F = 9u64.into();
        let b: F = 14u64.into();
        let m1 = a * b;
        let a1: F = 9u64.into();
        let b1: F = 14u64.into();
        let m2 = a1 * b1;
        let m3 = a1 * b1;
        let a2: F = 9u64.into();
        let b2: F = 14u64.into();
        let m4 = a2 * b2;
        assert_eq!(m1, m2);
        assert_eq!(m1, m3);
        assert_eq!(m1, m4);
    }

    #[test]
    fn assign_ops_with_refs_and_values() {
        let mut x: F = 7u64.into();
        let y: F = 8u64.into();
        x += y;
        assert_eq!(x, 15u64.into());
        let mut x: F = 7u64.into();
        let y: F = 8u64.into();
        x.add_assign(&y);
        assert_eq!(x, 15u64.into());
        let mut x: F = 20u64.into();
        let y: F = 6u64.into();
        x -= y;
        assert_eq!(x, 14u64.into());
        let mut x: F = 20u64.into();
        let y: F = 6u64.into();
        x.sub_assign(&y);
        assert_eq!(x, 14u64.into());
        let mut x: F = 5u64.into();
        let y: F = 9u64.into();
        x *= y;
        assert_eq!(x, 45u64.into());
        let mut x: F = 5u64.into();
        let y: F = 9u64.into();
        x.mul_assign(&y);
        assert_eq!(x, 45u64.into());
    }

    #[test]
    fn negation_properties() {
        let a: F = 12345u64.into();
        let zero = F::zero();
        assert_eq!(a + (-a), zero);
        assert_eq!(-(-a), a);
        assert_eq!(-zero, zero);
    }

    #[test]
    fn inversion_properties() {
        let a: F = 777u64.into();
        let inv = a.inv().expect("a should be invertible (non-zero)");
        assert_eq!(a * inv, F::one());
        let zero = F::zero();
        assert!(zero.inv().is_none());
    }

    #[test]
    #[should_panic]
    fn division_properties_and_errors() {
        let a: F = 9876u64.into();
        let b: F = 543u64.into();
        let q = a / b;
        assert_eq!(q * b, a);
        let c: F = 17u64.into();
        let bc = b * c;
        let left = a / bc;
        let right = (a / b) / c;
        assert_eq!(left, right);
        let _ = a / F::zero();
    }

    #[test]
    fn ring_identities() {
        let a: F = 3u64.into();
        let b: F = 5u64.into();
        let c: F = 7u64.into();
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
        let v: Vec<F> = vec![1u64.into(), 2u64.into(), 3u64.into(), 4u64.into()];
        let sum1: F = v.iter().cloned().sum();
        let sum2: F = v.iter().sum();
        let expected_sum: F = 10u64.into();
        assert_eq!(sum1, expected_sum);
        assert_eq!(sum2, expected_sum);

        let prod1: F = v.iter().product();
        // owned-product is not implemented; verify an equivalent fold
        let prod2: F = v.iter().fold(F::one(), |acc, x| acc * x);
        let expected_prod: F = (2 * 3 * 4).into();
        assert_eq!(prod1, expected_prod);
        assert_eq!(prod2, expected_prod);

        // empty iterators: define behavior as neutral elements
        let empty: Vec<F> = vec![];
        let sum_empty: F = empty.iter().cloned().sum();
        assert_eq!(sum_empty, F::zero());
        let prod_empty: F = empty.iter().product();
        assert_eq!(prod_empty, F::one());
    }
}

#[cfg(test)]
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
