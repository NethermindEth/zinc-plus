use crypto_primitives::{
    FromWithConfig, IntoWithConfig, PrimeField, crypto_bigint_monty::MontyField,
    crypto_bigint_uint::Uint,
};
use num_traits::CheckedMul;
use std::ops::MulAssign;

use crate::{
    from_ref::FromRef, inner_transparent_field::InnerTransparentField, mul_by_scalar::MulByScalar,
    projectable_to_field::ProjectableToField,
};

impl<const LIMBS: usize> MulByScalar<&Self> for MontyField<LIMBS> {
    fn mul_by_scalar(&self, rhs: &Self) -> Option<Self> {
        self.checked_mul(rhs)
    }
}

impl<const LIMBS: usize> FromRef<Self> for MontyField<LIMBS> {
    fn from_ref(value: &Self) -> Self {
        value.clone()
    }
}

impl<T, const LIMBS: usize> ProjectableToField<MontyField<LIMBS>> for T
where
    MontyField<LIMBS>: for<'a> FromWithConfig<&'a T>,
{
    fn prepare_projection(
        sampled_value: &MontyField<LIMBS>,
    ) -> impl Fn(&Self) -> MontyField<LIMBS> + Send + Sync + 'static {
        let config = sampled_value.cfg().clone();
        move |value: &T| value.into_with_cfg(&config)
    }
}

impl<const LIMBS: usize> InnerTransparentField for MontyField<LIMBS> {
    fn add_inner(lhs: &Self::Inner, rhs: &Self::Inner, config: &Self::Config) -> Self::Inner {
        Uint::new(
            lhs.inner()
                .add_mod(rhs.inner(), config.modulus().as_nz_ref()),
        )
    }

    fn sub_inner(lhs: &Self::Inner, rhs: &Self::Inner, config: &Self::Config) -> Self::Inner {
        Uint::new(
            lhs.inner()
                .sub_mod(rhs.inner(), config.modulus().as_nz_ref()),
        )
    }

    fn mul_assign_by_inner(&mut self, rhs: &Self::Inner) {
        let rhs: Self = Self::new_unchecked(*rhs, self.cfg());

        self.mul_assign(rhs);
    }
}

#[cfg(test)]
#[allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
mod prop_tests {
    use crypto_bigint::U256;
    use crypto_primitives::{
        FromWithConfig, IntoWithConfig, PrimeField,
        crypto_bigint_monty::{F256, MontyField},
        crypto_bigint_uint::Uint,
    };
    use proptest::prelude::*;

    use crate::inner_transparent_field::InnerTransparentField;

    const LIMBS: usize = 4;
    const MODULUS: &str = "00dca94d8a1ecce3b6e8755d8999787d0524d8ca1ea755e7af84fb646fa31f27";
    type F = F256;

    fn get_dyn_config(hex_modulus: &str) -> <MontyField<LIMBS> as PrimeField>::Config {
        let modulus = Uint::new(
            crypto_bigint::Uint::<LIMBS>::from_str_radix_vartime(hex_modulus, 16).unwrap(),
        );
        MontyField::make_cfg(&modulus).expect("Failed to create field config")
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
            let u = Uint::<{ U256::LIMBS }>::from(x);
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
            let u: Uint<LIMBS> = Uint::from(x);
            let g_from_uint: F = u.into_with_cfg(&cfg);
            let g_direct: F = x.into_with_cfg(&cfg);
            prop_assert_eq!(g_from_uint, g_direct);
        }

        #[test]
        fn prop_inner_ops_match_normal_ops((x, y) in any::<(u128, u128)>()) {
            let cfg = get_dyn_config(MODULUS);
            let a: F = x.into_with_cfg(&cfg);
            let b: F = y.into_with_cfg(&cfg);
            prop_assert_eq!(F::add_inner(a.inner(), b.inner(), &cfg), (a.clone() + b.clone()).into_inner());
            prop_assert_eq!(F::sub_inner(a.inner(), b.inner(), &cfg), (a.clone() - b.clone()).into_inner());

            let mut res = a.clone();
            res.mul_assign_by_inner(b.inner());
            prop_assert_eq!(res, a * b);
        }
    }
}
