use crypto_primitives::{
    ProjectElementWithConfig, SetConfig,
    crypto_bigint_monty::{MontyField, MontyFieldElement},
};

use crate::{from_ref::FromRef, projectable_to_field::ProjectableToField};

impl<const LIMBS: usize> FromRef<Self> for MontyFieldElement<LIMBS> {
    fn from_ref(value: &Self) -> Self {
        *value
    }
}

impl<T, const LIMBS: usize> ProjectableToField<MontyField<LIMBS>> for T
where
    MontyField<LIMBS>: ProjectElementWithConfig<T>,
{
    fn prepare_projection(
        cfg: &MontyField<LIMBS>,
        _sampled_value: &<MontyField<LIMBS> as SetConfig>::Element,
    ) -> impl Fn(&Self) -> <MontyField<LIMBS> as SetConfig>::Element {
        let cfg = *cfg;
        move |value: &T| cfg.project(value)
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
        BaseFieldConfig, ProjectElementWithConfig, SemiringConfig, crypto_bigint_monty::MontyField,
        crypto_bigint_uint::Uint,
    };
    use proptest::prelude::*;
    use std::str::FromStr;

    const LIMBS: usize = 4;
    const MODULUS: &str = "00dca94d8a1ecce3b6e8755d8999787d0524d8ca1ea755e7af84fb646fa31f27";

    fn get_dyn_config(hex_modulus: &str) -> MontyField<LIMBS> {
        let modulus =
            Uint::from_str(&format!("0x{hex_modulus}")).expect("Invalid modulus hex string");
        MontyField::new(&modulus).expect("Failed to create field config")
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
        #[cfg_attr(miri, ignore)] // long running
        fn prop_from_unsigned_matches_sum_of_bits(x in any_u128()) {
            let cfg = get_dyn_config(MODULUS);
            let f = cfg.project(&x);
            let mut acc = cfg.zero();
            for i in 0..128 {
                if (x >> i) & 1 == 1 {
                    let bit = cfg.mul(&cfg.project(&1u64), &cfg.project(&(1u64 << i.min(63))));
                    cfg.add_assign(&mut acc, &bit);
                }
            }
            let u = Uint::<{ U256::LIMBS }>::from(x);
            let g2 = cfg.project(&u);
            prop_assert_eq!(f, g2);
        }

        #[test]
        #[cfg_attr(miri, ignore)] // long running
        fn prop_from_signed_is_neg_of_abs_when_negative(x in any_i128()) {
            let cfg = get_dyn_config(MODULUS);
            let f = cfg.project(&x);
            let abs = x.unsigned_abs();
            let g_abs = cfg.project(&abs);
            if x < 0 {
                prop_assert_eq!(cfg.add(&f, &g_abs), cfg.zero());
            } else {
                prop_assert_eq!(f, g_abs);
            }
        }

        #[test]
        #[cfg_attr(miri, ignore)] // long running
        fn prop_from_bool_is_identity(b in any_bool()) {
            let cfg = get_dyn_config(MODULUS);
            let f = cfg.project(&b);
            prop_assert_eq!(f, if b { cfg.one() } else { cfg.zero() });
        }

        #[test]
        #[cfg_attr(miri, ignore)] // long running
        fn prop_from_uint_roundtrip_through_uint(x in any_u128()) {
            let cfg = get_dyn_config(MODULUS);
            let u: Uint<LIMBS> = Uint::from(x);
            let g_from_uint = cfg.project(&u);
            let g_direct = cfg.project(&x);
            prop_assert_eq!(g_from_uint, g_direct);
        }
    }
}
