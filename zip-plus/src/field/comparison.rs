use std::cmp::Ordering;
use crypto_bigint::modular::ConstMontyParams;
use crypto_bigint::subtle::{Choice, ConstantTimeEq};

use crate::field::{ConstMontyField};

impl<MOD: ConstMontyParams<LIMBS>, const LIMBS: usize> ConstantTimeEq for ConstMontyField<MOD, LIMBS> {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.0.ct_eq(&other.0)
    }
}

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> PartialOrd for ConstMontyField<Mod, LIMBS> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> Ord for ConstMontyField<Mod, LIMBS> {
    fn cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(self.0.as_montgomery(), other.0.as_montgomery())
    }
}

#[cfg(test)]
mod tests {
    use crypto_bigint::{U128, const_monty_params};

    use super::*;

    const_monty_params!(ModP, U128, "7fffffffffffffffffffffffffffffff");
    type F = ConstMontyField<ModP, { U128::LIMBS }>;

    #[test]
    fn const_time_eq_and_order() {
        let a: F = 10u64.into();
        let b: F = 10u64.into();
        let c: F = 11u64.into();
        assert_eq!(a.ct_eq(&b).unwrap_u8(), 1);
        assert_eq!(a.ct_eq(&c).unwrap_u8(), 0);
        assert!(a.partial_cmp(&c).is_some());
    }
}
