use crypto_bigint::BoxedUint;
use crypto_primitives::{
    FromWithConfig, IntoWithConfig, PrimeField, crypto_bigint_boxed_monty::BoxedMontyField,
    crypto_bigint_uint::Uint,
};

use crate::{
    from_ref::FromRef, mul_by_scalar::MulByScalar, projectable_to_field::ProjectableToField,
};

impl MulByScalar<&Self> for BoxedMontyField {
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn mul_by_scalar<const CHECK: bool>(&self, rhs: &Self) -> Option<Self> {
        // Field operations cannot overflow
        Some(self * rhs)
    }
}

impl FromRef<Self> for BoxedMontyField {
    fn from_ref(value: &Self) -> Self {
        value.clone()
    }
}

impl<const LIMBS: usize> FromRef<Uint<LIMBS>> for BoxedUint {
    #[inline]
    fn from_ref(value: &Uint<LIMBS>) -> Self {
        value.inner().into()
    }
}

impl<T> ProjectableToField<BoxedMontyField> for T
where
    BoxedMontyField: for<'a> FromWithConfig<&'a T>,
{
    fn prepare_projection(
        sampled_value: &BoxedMontyField,
    ) -> impl Fn(&Self) -> BoxedMontyField + Send + Sync + 'static {
        let config = sampled_value.cfg().clone();
        move |value: &T| value.into_with_cfg(&config)
    }
}

#[cfg(test)]
#[allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
mod prop_tests {
    use crypto_bigint::{BoxedUint, U256};
    use crypto_primitives::{
        FromWithConfig, IntoWithConfig, PrimeField, crypto_bigint_boxed_monty::BoxedMontyField,
    };
    use proptest::prelude::*;

    const MODULUS: &str = "00dca94d8a1ecce3b6e8755d8999787d0524d8ca1ea755e7af84fb646fa31f27";
    type F = BoxedMontyField;

    fn get_dyn_config(hex_modulus: &str) -> <BoxedMontyField as PrimeField>::Config {
        let modulus =
            BoxedUint::from_str_radix_vartime(hex_modulus, 16).expect("Invalid modulus hex string");
        BoxedMontyField::make_cfg(&modulus).expect("Failed to create field config")
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
            let cfg = get_dyn_config(MODULUS);
            let f: F = x.into_with_cfg(&cfg);
            let mut acc: F = F::zero_with_cfg(&cfg);
            for i in 0..128 {
                if (x >> i) & 1 == 1 { acc += F::from_with_cfg(1u64, &cfg) * F::from_with_cfg(1u64 << i.min(63), &cfg); }
            }
            let u = crypto_bigint::Uint::<{ U256::LIMBS }>::from(x);
            let g2: F = u.into_with_cfg(&cfg);
            prop_assert_eq!(f, g2);
        }

        #[test]
        fn prop_from_signed_is_neg_of_abs_when_negative(x in any_i128()) {
            let cfg = get_dyn_config(MODULUS);
            let f: F = x.into_with_cfg(&cfg);
            let abs = x.unsigned_abs();
            let g_abs = abs.into_with_cfg(&cfg);
            if x < 0 {
                prop_assert_eq!(f + g_abs, F::zero_with_cfg(&cfg));
            } else {
                prop_assert_eq!(f, g_abs);
            }
        }

        #[test]
        fn prop_from_bool_is_identity(b in any_bool()) {
            let cfg = get_dyn_config(MODULUS);
            let f: F = b.into_with_cfg(&cfg);
            prop_assert_eq!(f, if b { F::one_with_cfg(&cfg) } else { F::zero_with_cfg(&cfg) });
        }

        #[test]
        fn prop_from_uint_roundtrip_through_uint(x in any_u128()) {
            let cfg = get_dyn_config(MODULUS);
            let u: crypto_bigint::Uint<{ U256::LIMBS }> = crypto_bigint::Uint::from(x);
            let g_from_uint: F = u.into_with_cfg(&cfg);
            let g_direct: F = x.into_with_cfg(&cfg);
            prop_assert_eq!(g_from_uint, g_direct);
        }

        #[test]
        fn prop_from_with_cfg_generic_matches_owned(x in any::<u64>()) {
            let cfg = get_dyn_config(MODULUS);
            let a: F = x.into_with_cfg(&cfg);
            let b: F = F::from_with_cfg(&x, &cfg);
            prop_assert_eq!(a, b);
        }
    }
}
