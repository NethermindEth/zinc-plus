use crypto_bigint::modular::{ConstMontyForm, ConstMontyParams};
use crypto_primitives::{
    crypto_bigint_const_monty::ConstMontyField, crypto_bigint_int::Int, crypto_bigint_uint::Uint,
};
use std::ops::MulAssign;

use crate::{
    from_ref::FromRef, inner_transparent_field::InnerTransparentField, mul_by_scalar::MulByScalar,
    projectable_to_field::ProjectableToField,
};

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> MulByScalar<&Self>
    for ConstMontyField<Mod, LIMBS>
{
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn mul_by_scalar<const CHECK: bool>(&self, rhs: &Self) -> Option<Self> {
        // Multiplication cannot overflow
        Some(self * rhs)
    }
}

macro_rules! impl_from_primitive_ref {
    ($($t:ty),* $(,)?) => {
        $(
            impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> FromRef<$t> for ConstMontyField<Mod, LIMBS> {
                #![allow(clippy::arithmetic_side_effects)]
                fn from_ref(value: &$t) -> Self {
                    Self::from(*value)
                }
            }
        )*
    };
}
impl_from_primitive_ref!(u8, u16, u32, u64, u128);

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> FromRef<Uint<LIMBS>>
    for ConstMontyForm<Mod, LIMBS>
{
    fn from_ref(value: &Uint<LIMBS>) -> Self {
        ConstMontyForm::new(value.inner())
    }
}

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> FromRef<Self>
    for ConstMontyField<Mod, LIMBS>
{
    fn from_ref(value: &Self) -> Self {
        *value
    }
}

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize, const LIMBS2: usize>
    ProjectableToField<ConstMontyField<Mod, LIMBS>> for Int<LIMBS2>
{
    fn prepare_projection(
        _sampled_value: &ConstMontyField<Mod, LIMBS>,
    ) -> impl Fn(&Self) -> ConstMontyField<Mod, LIMBS> + 'static {
        // No need to read anything
        |value: &Int<LIMBS2>| value.into()
    }
}

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> InnerTransparentField
    for ConstMontyField<Mod, LIMBS>
{
    fn add_inner(lhs: &Self::Inner, rhs: &Self::Inner, _config: &Self::Config) -> Self::Inner {
        Uint::new(
            lhs.inner()
                .add_mod(rhs.inner(), Mod::PARAMS.modulus().as_nz_ref()),
        )
    }

    fn sub_inner(lhs: &Self::Inner, rhs: &Self::Inner, _config: &Self::Config) -> Self::Inner {
        Uint::new(
            lhs.inner()
                .sub_mod(rhs.inner(), Mod::PARAMS.modulus().as_nz_ref()),
        )
    }

    fn mul_assign_by_inner(&mut self, rhs: &Self::Inner) {
        let rhs: Self = Self::new_unchecked(*rhs);

        self.mul_assign(rhs);
    }
}

#[cfg(test)]
#[allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
mod tests {
    use crate::projectable_to_field::ProjectableToField;

    use super::*;

    use crypto_bigint::{U128, U256, const_monty_params};
    use crypto_primitives::{crypto_bigint_const_monty::F256, crypto_bigint_int::Int};
    use num_traits::{One, Zero};
    use proptest::prelude::*;

    const_monty_params!(
        ModQ,
        U256,
        "00dca94d8a1ecce3b6e8755d8999787d0524d8ca1ea755e7af84fb646fa31f27"
    );
    type F = F256<ModQ>;

    #[test]
    fn prepare_projection_for_int() {
        // Create a sample field element and an Int value
        let sampled = F::from(5_u64);

        let projection_fn = Int::<{ U128::LIMBS }>::prepare_projection(&sampled);

        let int_value = Int::<{ U128::LIMBS }>::from(10_i64);
        let result = projection_fn(&int_value);
        assert_eq!(result, F::from(&int_value));
        assert_eq!(result, F::from(10_u64));

        let int_value = Int::<{ U128::LIMBS }>::from(-7_i64);
        let result = projection_fn(&int_value);
        assert_eq!(result, F::from(&int_value));
        assert_eq!(result + F::from(7_u64), F::zero());
    }

    fn any_u128() -> impl Strategy<Value = u128> {
        any::<u128>()
    }
    fn any_i128() -> impl Strategy<Value = i128> {
        any::<i128>()
    }
    fn any_bool() -> impl Strategy<Value = bool> {
        any::<bool>()
    }

    proptest! {
        #[test]
        fn prop_from_unsigned_matches_sum_of_bits(x in any_u128()) {
            let f = F::from(x);
            let mut acc = F::zero();
            for i in 0..128 {
                if (x >> i) & 1 == 1 { acc += F::from(1u64) * F::from(1u64 << i.min(63)); }
            }
            let u = Uint::<{ U256::LIMBS }>::from(x);
            let g2: F = F::from(u);
            prop_assert_eq!(f, g2);
        }

        #[test]
        fn prop_from_signed_is_neg_of_abs_when_negative(x in any_i128()) {
            let f = F::from(x);
            let abs = x.unsigned_abs();
            let g_abs = F::from(abs);
            if x < 0 {
                prop_assert_eq!(f + g_abs, F::zero());
            } else {
                prop_assert_eq!(f, g_abs);
            }
        }

        #[test]
        fn prop_from_bool_is_identity(b in any_bool()) {
            let f = F::from(b);
            prop_assert_eq!(f, if b { F::one() } else { F::zero() });
        }

        #[test]
        fn prop_from_uint_roundtrip_through_uint(x in any_u128()) {
            let u: Uint<{ U256::LIMBS }> = Uint::from(x);
            let g_from_uint: F = u.into();
            let g_direct: F = F::from(x);
            prop_assert_eq!(g_from_uint, g_direct);
        }

        #[test]
        fn prop_from_ref_generic_matches_owned(x in any::<u64>()) {
            let a: F = F::from(x);
            let b: F = F::from(&x);
            prop_assert_eq!(a, b);
        }

        #[test]
        fn prop_inner_ops_match_normal_ops((x, y) in any::<(u128, u128)>()) {
            let a: F = x.into();
            let b: F = y.into();
            prop_assert_eq!(F::add_inner(a.inner(), b.inner(), &()), (a + b).into_inner());
            prop_assert_eq!(F::sub_inner(a.inner(), b.inner(), &()), (a - b).into_inner());

            let mut res = a;
            res.mul_assign_by_inner(b.inner());
            prop_assert_eq!(res, a * b);
        }
    }
}
