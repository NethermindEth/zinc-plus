use crate::field::ConstMontyField;
use crypto_bigint::modular::{ConstMontyForm, ConstMontyParams};
use num_traits::{ConstOne, ConstZero, One, Zero};

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> ConstZero for ConstMontyField<Mod, LIMBS> {
    const ZERO: Self = Self(ConstMontyForm::ZERO);
}

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> ConstOne for ConstMontyField<Mod, LIMBS> {
    const ONE: Self = Self(ConstMontyForm::ONE);
}

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> Zero for ConstMontyField<Mod, LIMBS> {
    #[inline]
    fn zero() -> Self {
        Self::ZERO
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self == &Self::ZERO
    }
}

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> One for ConstMontyField<Mod, LIMBS> {
    fn one() -> Self {
        Self::ONE
    }
}

#[cfg(test)]
mod tests {
    use crypto_bigint::{U128, const_monty_params};
    use num_traits::{One, Zero};

    use super::*;

    // Define a small prime modulus for tests (copying style from Zinc tests where
    // modulus ~ 128-bit)
    const_monty_params!(ModP, U128, "7fffffffffffffffffffffffffffffff");

    type F = ConstMontyField<ModP, { U128::LIMBS }>;

    #[test]
    fn zero_one_basics() {
        let z = F::zero();
        assert!(z.is_zero());
        let o = F::one();
        assert!(!o.is_zero());
        assert_ne!(z, o);
    }

    #[test]
    fn from_bool_matches_one_zero() {
        let t: F = true.into();
        let f: F = false.into();
        assert_eq!(t, F::one());
        assert_eq!(f, F::zero());
    }
}
