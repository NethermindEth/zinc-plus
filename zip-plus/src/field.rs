#![allow(non_snake_case)]

pub mod arithmetic;
pub mod comparison;
pub mod conversion;
pub mod numeric;
mod utils;

use std::fmt;
use crypto_bigint::modular::{ConstMontyForm, ConstMontyParams};
use crypto_bigint::rand_core::TryRngCore;
use crypto_bigint::Random;
use std::fmt::{Display, Formatter};

use crypto_primitives::{ConstRing, Field, IntRing, PrimeField, Ring};
use crate::utils::WORD_FACTOR;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct ConstMontyField<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize>(pub ConstMontyForm<Mod, LIMBS>);

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> Display for ConstMontyField<Mod, LIMBS> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{} (mod {})", self.0.retrieve(), Mod::PARAMS.modulus())
    }
}

impl<MOD: ConstMontyParams<LIMBS>, const LIMBS: usize> Random for ConstMontyField<MOD, LIMBS> {
    fn try_random<R: TryRngCore + ?Sized>(rng: &mut R) -> Result<Self, R::Error> {
        Ok(Self(ConstMontyForm::try_random(rng)?))
    }
}

impl<MOD: ConstMontyParams<LIMBS>, const LIMBS: usize> Ring for ConstMontyField<MOD, LIMBS> {}

impl<MOD: ConstMontyParams<LIMBS>, const LIMBS: usize> ConstRing for ConstMontyField<MOD, LIMBS> {}

impl<MOD: ConstMontyParams<LIMBS>, const LIMBS: usize> IntRing for ConstMontyField<MOD, LIMBS> {}

impl<MOD: ConstMontyParams<LIMBS>, const LIMBS: usize> Field for ConstMontyField<MOD, LIMBS> {}

impl<MOD: ConstMontyParams<LIMBS>, const LIMBS: usize> PrimeField for ConstMontyField<MOD, LIMBS> {
    type Inner = ConstMontyForm<MOD, LIMBS>;
    const MODULUS: Self::Inner = ConstMontyForm::<MOD, LIMBS>::new(MOD::PARAMS.modulus().as_ref());

    #[inline(always)]
    fn new_unchecked(value: Self::Inner) -> Self {
        Self(value)
    }

    #[inline(always)]
    fn inner(&self) -> &Self::Inner {
        &self.0
    }
}

pub type F64<MOD> = ConstMontyField<MOD, { WORD_FACTOR }>;
pub type F128<MOD> = ConstMontyField<MOD, { 2 * WORD_FACTOR }>;
pub type F192<MOD> = ConstMontyField<MOD, { 3 * WORD_FACTOR }>;
pub type F256<MOD> = ConstMontyField<MOD, { 4 * WORD_FACTOR }>;
pub type F320<MOD> = ConstMontyField<MOD, { 5 * WORD_FACTOR }>;
pub type F384<MOD> = ConstMontyField<MOD, { 6 * WORD_FACTOR }>;
pub type F448<MOD> = ConstMontyField<MOD, { 7 * WORD_FACTOR }>;
pub type F512<MOD> = ConstMontyField<MOD, { 8 * WORD_FACTOR }>;
pub type F576<MOD> = ConstMontyField<MOD, { 9 * WORD_FACTOR }>;
pub type F640<MOD> = ConstMontyField<MOD, { 10 * WORD_FACTOR }>;
pub type F704<MOD> = ConstMontyField<MOD, { 11 * WORD_FACTOR }>;
pub type F768<MOD> = ConstMontyField<MOD, { 12 * WORD_FACTOR }>;
pub type F832<MOD> = ConstMontyField<MOD, { 13 * WORD_FACTOR }>;
pub type F896<MOD> = ConstMontyField<MOD, { 14 * WORD_FACTOR }>;
pub type F960<MOD> = ConstMontyField<MOD, { 15 * WORD_FACTOR }>;
pub type F1024<MOD> = ConstMontyField<MOD, { 16 * WORD_FACTOR }>;
pub type F1280<MOD> = ConstMontyField<MOD, { 20 * WORD_FACTOR }>;
pub type F1536<MOD> = ConstMontyField<MOD, { 24 * WORD_FACTOR }>;
pub type F1792<MOD> = ConstMontyField<MOD, { 28 * WORD_FACTOR }>;
pub type F2048<MOD> = ConstMontyField<MOD, { 32 * WORD_FACTOR }>;
pub type F3072<MOD> = ConstMontyField<MOD, { 48 * WORD_FACTOR }>;
pub type F3584<MOD> = ConstMontyField<MOD, { 56 * WORD_FACTOR }>;
pub type F4096<MOD> = ConstMontyField<MOD, { 64 * WORD_FACTOR }>;
pub type F4224<MOD> = ConstMontyField<MOD, { 66 * WORD_FACTOR }>;
pub type F4352<MOD> = ConstMontyField<MOD, { 68 * WORD_FACTOR }>;
pub type F6144<MOD> = ConstMontyField<MOD, { 96 * WORD_FACTOR }>;
pub type F8192<MOD> = ConstMontyField<MOD, { 128 * WORD_FACTOR }>;
pub type F16384<MOD> = ConstMontyField<MOD, { 256 * WORD_FACTOR }>;
pub type F32768<MOD> = ConstMontyField<MOD, { 512 * WORD_FACTOR }>;

#[cfg(test)]
mod tests {
    use crypto_bigint::{const_monty_params, U128};

    use super::*;

    const_monty_params!(ModP, U128, "7fffffffffffffffffffffffffffffff");

    type F = ConstMontyField<ModP, { U128::LIMBS }>;

    #[test]
    fn basic_add_smoke() {
        let a: F = 123u64.into();
        let b: F = 456u64.into();
        assert_eq!(a + b, F::from(579u64));
    }
}
