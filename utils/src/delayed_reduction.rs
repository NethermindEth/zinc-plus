//! Delayed modular reduction helpers for fixed 4-limb Montgomery fields.
//!
//! This module is intentionally narrow: it supports summing Montgomery-form
//! field elements into a 5-limb accumulator, then reducing once with Barrett
//! reduction. The limb routines are adapted from Spartan2's MIT-licensed
//! `big_num` helpers.

use crypto_bigint::modular::{ConstMontyForm, ConstMontyParams, MontyForm};
use crypto_primitives::{
    PrimeField, crypto_bigint_const_monty::ConstMontyField, crypto_bigint_monty::MontyField,
    crypto_bigint_uint::Uint,
};
use num_traits::Zero;

/// Barrett reduction parameters modulo a 4-limb prime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BarrettReductionParams {
    /// The 4-limb prime modulus in little-endian limb order.
    pub modulus: [u64; 4],

    /// `floor(2^512 / MODULUS)`, stored in little-endian limb order.
    pub mu: [u64; 5],
}

impl BarrettReductionParams {
    #[inline(always)]
    pub const fn new(modulus: [u64; 4]) -> Self {
        Self {
            modulus,
            mu: compute_barrett_mu(modulus),
        }
    }
}

/// Field types that expose reduced Montgomery-form limbs.
pub trait MontgomeryLimbs: PrimeField<Inner = Uint<4>> + Sized {
    /// Construct a field element from reduced Montgomery-form limbs.
    fn from_montgomery_limbs(limbs: [u64; 4], cfg: &Self::Config) -> Self;

    /// Borrow the field element's Montgomery-form limbs.
    fn montgomery_limbs(&self) -> &[u64; 4];

    /// Return Barrett reduction parameters for this field configuration.
    fn barrett_reduction_params(cfg: &Self::Config) -> BarrettReductionParams;
}

/// Accumulator trait for delayed modular reduction.
pub trait DelayedModularReduction<F>: Zero + Clone + Send + Sync
where
    F: PrimeField,
{
    fn add(&mut self, value: &F);
    fn reduce(self, cfg: &F::Config, params: &BarrettReductionParams) -> F;
}

/// Field product-sum backend for delayed modular reduction-aware dot products.
pub trait DelayedFieldProductSum: PrimeField + Sized {
    /// Compute `zero + sum_i lhs[i] * rhs[i]`.
    ///
    /// The caller is responsible for enforcing equal slice lengths.
    fn delayed_sum_of_products(lhs: &[Self], rhs: &[Self], zero: Self) -> Self;
}

impl<F> DelayedModularReduction<F> for Uint<5>
where
    F: MontgomeryLimbs + Send + Sync,
{
    #[inline(always)]
    fn add(&mut self, value: &F) {
        let acc = self.as_mut_words();
        let rhs = value.montgomery_limbs();
        let mut carry = 0u64;
        let mut i = 0;
        while i < 4 {
            let (sum, c0) = acc[i].overflowing_add(rhs[i]);
            let (sum, c1) = sum.overflowing_add(carry);
            acc[i] = sum;
            carry = (c0 as u64) + (c1 as u64);
            i += 1;
        }

        let old_hi = acc[4];
        acc[4] = acc[4].wrapping_add(carry);
        debug_assert!(
            acc[4] >= old_hi,
            "Uint<5> delayed accumulator overflowed high limb"
        );
    }

    #[inline(always)]
    fn reduce(self, cfg: &F::Config, params: &BarrettReductionParams) -> F {
        let acc = self.as_words();
        F::from_montgomery_limbs(barrett_reduce_5(acc, params), cfg)
    }
}

impl<const LIMBS: usize> DelayedFieldProductSum for MontyField<LIMBS> {
    fn delayed_sum_of_products(lhs: &[Self], rhs: &[Self], zero: Self) -> Self {
        if lhs.is_empty() {
            return zero;
        }

        let leading_zeros = zero.cfg().modulus().as_ref().leading_zeros();
        if !lincomb_has_product_sum_headroom(leading_zeros, lhs.len()) {
            return naive_sum_of_products(lhs, rhs, zero);
        }

        let lhs_forms: Vec<MontyForm<LIMBS>> =
            lhs.iter().cloned().map(|value| value.into()).collect();
        let rhs_forms: Vec<MontyForm<LIMBS>> =
            rhs.iter().cloned().map(|value| value.into()).collect();
        let products: Vec<(&MontyForm<LIMBS>, &MontyForm<LIMBS>)> =
            lhs_forms.iter().zip(&rhs_forms).collect();

        MontyField::new(MontyForm::lincomb_vartime(&products)) + zero
    }
}

impl<Mod, const LIMBS: usize> DelayedFieldProductSum for ConstMontyField<Mod, LIMBS>
where
    Mod: ConstMontyParams<LIMBS>,
{
    fn delayed_sum_of_products(lhs: &[Self], rhs: &[Self], zero: Self) -> Self {
        if lhs.is_empty() {
            return zero;
        }

        let leading_zeros = Mod::PARAMS.modulus().as_ref().leading_zeros();
        if !lincomb_has_product_sum_headroom(leading_zeros, lhs.len()) {
            return naive_sum_of_products(lhs, rhs, zero);
        }

        let products: Vec<(ConstMontyForm<Mod, LIMBS>, ConstMontyForm<Mod, LIMBS>)> = lhs
            .iter()
            .cloned()
            .zip(rhs.iter().cloned())
            .map(|(left, right)| (left.into(), right.into()))
            .collect();

        ConstMontyField::from(ConstMontyForm::lincomb(&products)) + zero
    }
}

#[inline(always)]
fn lincomb_has_product_sum_headroom(leading_zeros: u32, len: usize) -> bool {
    len > 1 && leading_zeros > 0
}

#[allow(clippy::arithmetic_side_effects)]
fn naive_sum_of_products<F: PrimeField>(lhs: &[F], rhs: &[F], zero: F) -> F {
    lhs.iter()
        .zip(rhs)
        .fold(zero, |acc, (left, right)| acc + left.clone() * right)
}

impl<Mod> MontgomeryLimbs for ConstMontyField<Mod, 4>
where
    Mod: ConstMontyParams<4>,
{
    #[inline(always)]
    fn from_montgomery_limbs(limbs: [u64; 4], _cfg: &Self::Config) -> Self {
        Self::new_unchecked(Uint::<4>::from_words(limbs))
    }

    #[inline(always)]
    fn montgomery_limbs(&self) -> &[u64; 4] {
        self.inner().as_words()
    }

    #[inline(always)]
    fn barrett_reduction_params(_cfg: &Self::Config) -> BarrettReductionParams {
        BarrettReductionParams::new(Uint::<4>::new(*Mod::PARAMS.modulus().as_ref()).to_words())
    }
}

impl MontgomeryLimbs for MontyField<4> {
    #[inline(always)]
    fn from_montgomery_limbs(limbs: [u64; 4], cfg: &Self::Config) -> Self {
        Self::new_unchecked_with_cfg(Uint::<4>::from_words(limbs), cfg)
    }

    #[inline(always)]
    fn montgomery_limbs(&self) -> &[u64; 4] {
        self.inner().as_words()
    }

    #[inline(always)]
    fn barrett_reduction_params(cfg: &Self::Config) -> BarrettReductionParams {
        BarrettReductionParams::new(Uint::<4>::new(cfg.modulus().get()).to_words())
    }
}

/// Barrett reduction for a 5-limb value modulo a 4-limb modulus.
///
/// This uses the 5-limb remainder path, which is required for moduli near
/// `2^256` such as the secp256k1 base prime.
#[inline(always)]
pub fn barrett_reduce_5(c: &[u64; 5], params: &BarrettReductionParams) -> [u64; 4] {
    let q1 = [c[3], c[4]];
    let q2 = mul_2x5_to_7(&q1, &params.mu);
    let q3 = [q2[5], q2[6]];

    let r1 = *c;
    let r2 = mul_2x4_lo5(&q3, &params.modulus);
    let mut r = sub::<5>(&r1, &r2);

    if r[4] != 0 || gte::<4>(&[r[0], r[1], r[2], r[3]], &params.modulus) {
        r = sub_5_4(&r, &params.modulus);
    }

    debug_assert!(
        r[4] == 0 && !gte::<4>(&[r[0], r[1], r[2], r[3]], &params.modulus),
        "Barrett reduction produced non-canonical result"
    );

    [r[0], r[1], r[2], r[3]]
}

#[inline(always)]
fn mul_2x5_to_7(a: &[u64; 2], b: &[u64; 5]) -> [u64; 7] {
    let mut result = [0u64; 7];
    for i in 0..2 {
        let mut carry = 0u128;
        for j in 0..5 {
            let prod = (a[i] as u128) * (b[j] as u128) + (result[i + j] as u128) + carry;
            result[i + j] = prod as u64;
            carry = prod >> 64;
        }
        result[i + 5] = carry as u64;
    }
    result
}

#[inline(always)]
fn mul_2x4_lo5(a: &[u64; 2], b: &[u64; 4]) -> [u64; 5] {
    let mut result = [0u64; 5];

    let mut carry = 0u128;
    for j in 0..4 {
        let prod = (a[0] as u128) * (b[j] as u128) + carry;
        result[j] = prod as u64;
        carry = prod >> 64;
    }
    result[4] = carry as u64;

    carry = 0;
    for j in 0..4 {
        let prod = (a[1] as u128) * (b[j] as u128) + (result[1 + j] as u128) + carry;
        result[1 + j] = prod as u64;
        carry = prod >> 64;
    }

    result
}

#[inline(always)]
const fn gte<const N: usize>(a: &[u64; N], b: &[u64; N]) -> bool {
    let mut i = N;
    while i > 0 {
        i -= 1;
        if a[i] > b[i] {
            return true;
        }
        if a[i] < b[i] {
            return false;
        }
    }
    true
}

#[inline(always)]
const fn sub<const N: usize>(a: &[u64; N], b: &[u64; N]) -> [u64; N] {
    let mut result = [0u64; N];
    let mut borrow = 0u64;
    let mut i = 0;
    while i < N {
        let (diff, b1) = a[i].overflowing_sub(b[i]);
        let (diff2, b2) = diff.overflowing_sub(borrow);
        result[i] = diff2;
        borrow = (b1 as u64) + (b2 as u64);
        i += 1;
    }
    result
}

#[inline(always)]
const fn sub_5_4(a: &[u64; 5], b: &[u64; 4]) -> [u64; 5] {
    let mut result = [0u64; 5];
    let mut borrow = 0u64;
    let mut i = 0;
    while i < 4 {
        let (diff, b1) = a[i].overflowing_sub(b[i]);
        let (diff2, b2) = diff.overflowing_sub(borrow);
        result[i] = diff2;
        borrow = (b1 as u64) + (b2 as u64);
        i += 1;
    }
    let (diff, _) = a[4].overflowing_sub(borrow);
    result[4] = diff;
    result
}

#[inline(always)]
const fn shl<const N: usize>(a: &[u64; N]) -> [u64; N] {
    let mut result = [0u64; N];
    let mut carry = 0u64;
    let mut i = 0;
    while i < N {
        let new_carry = a[i] >> 63;
        result[i] = (a[i] << 1) | carry;
        carry = new_carry;
        i += 1;
    }
    result
}

#[inline(always)]
const fn shr<const N: usize>(a: &[u64; N]) -> [u64; N] {
    let mut result = [0u64; N];
    let mut carry = 0u64;
    let mut i = N;
    while i > 0 {
        i -= 1;
        let new_carry = a[i] << 63;
        result[i] = (a[i] >> 1) | carry;
        carry = new_carry;
    }
    result
}

#[inline(always)]
const fn clz<const N: usize>(a: &[u64; N]) -> u32 {
    let mut i = N;
    let mut count = 0u32;
    while i > 0 {
        i -= 1;
        if a[i] != 0 {
            return count + a[i].leading_zeros();
        }
        count += 64;
    }
    count
}

pub const fn compute_barrett_mu(p: [u64; 4]) -> [u64; 5] {
    let mut dividend: [u64; 9] = [0, 0, 0, 0, 0, 0, 0, 0, 1];
    let divisor: [u64; 9] = [p[0], p[1], p[2], p[3], 0, 0, 0, 0, 0];
    let mut quotient: [u64; 5] = [0; 5];

    let dividend_clz = clz::<9>(&dividend);
    let divisor_clz = clz::<9>(&divisor);
    if divisor_clz <= dividend_clz {
        return quotient;
    }

    let shift_bits = divisor_clz - dividend_clz;
    let mut shifted_divisor = divisor;
    let whole_limbs = (shift_bits / 64) as usize;
    let rem_bits = shift_bits % 64;

    if whole_limbs > 0 {
        let mut i = 8;
        while i >= whole_limbs {
            shifted_divisor[i] = shifted_divisor[i - whole_limbs];
            if i == whole_limbs {
                break;
            }
            i -= 1;
        }
        let mut j = 0;
        while j < whole_limbs {
            shifted_divisor[j] = 0;
            j += 1;
        }
    }

    let mut i = 0;
    while i < rem_bits {
        shifted_divisor = shl::<9>(&shifted_divisor);
        i += 1;
    }

    let mut bit_pos = shift_bits;
    loop {
        if gte::<9>(&dividend, &shifted_divisor) {
            dividend = sub::<9>(&dividend, &shifted_divisor);
            let limb_idx = (bit_pos / 64) as usize;
            let bit_idx = bit_pos % 64;
            if limb_idx < 5 {
                quotient[limb_idx] |= 1u64 << bit_idx;
            }
        }

        if bit_pos == 0 {
            break;
        }
        bit_pos -= 1;
        shifted_divisor = shr::<9>(&shifted_divisor);
    }

    quotient
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_primitives::{FromWithConfig, crypto_bigint_monty::MontyField};

    type F = MontyField<4>;

    fn secp256k1_cfg() -> <F as PrimeField>::Config {
        let modulus = Uint::<4>::from_words([
            0xFFFF_FFFE_FFFF_FC2F,
            0xFFFF_FFFF_FFFF_FFFF,
            0xFFFF_FFFF_FFFF_FFFF,
            0xFFFF_FFFF_FFFF_FFFF,
        ]);
        F::make_cfg(&modulus).expect("secp256k1 base field prime is valid")
    }

    #[test]
    fn secp256k1_barrett_params_match_expected_modulus() {
        let cfg = secp256k1_cfg();
        assert_eq!(
            F::barrett_reduction_params(&cfg).modulus,
            [
                0xFFFF_FFFE_FFFF_FC2F,
                0xFFFF_FFFF_FFFF_FFFF,
                0xFFFF_FFFF_FFFF_FFFF,
                0xFFFF_FFFF_FFFF_FFFF,
            ],
        );
    }

    #[test]
    fn delayed_sum_matches_field_addition() {
        let cfg = secp256k1_cfg();
        let reduction_params = F::barrett_reduction_params(&cfg);
        let values: Vec<F> = (0..512)
            .map(|i| F::from_with_cfg(i as u64 + 1, &cfg))
            .collect();

        let mut expected = F::zero_with_cfg(&cfg);
        for value in &values {
            expected += value;
        }

        let mut acc = Uint::<5>::zero();
        for value in &values {
            <Uint<5> as DelayedModularReduction<F>>::add(&mut acc, value);
        }

        assert_eq!(
            <Uint<5> as DelayedModularReduction<F>>::reduce(acc, &cfg, &reduction_params),
            expected
        );
    }

    #[test]
    fn barrett_reduce_5_matches_uint_remainder_for_bounded_sum() {
        let cfg = secp256k1_cfg();
        let reduction_params = F::barrett_reduction_params(&cfg);
        let mut acc = Uint::<5>::zero();
        let max = -F::from_with_cfg(1u64, &cfg);
        for _ in 0..512 {
            <Uint<5> as DelayedModularReduction<F>>::add(&mut acc, &max);
        }

        let wide = acc;
        let modulus = Uint::<5>::from_words([
            reduction_params.modulus[0],
            reduction_params.modulus[1],
            reduction_params.modulus[2],
            reduction_params.modulus[3],
            0,
        ]);
        let expected = (wide % &modulus)
            .checked_resize::<4>()
            .expect("remainder fits in four limbs");

        let acc = acc.as_words();
        let reduced = barrett_reduce_5(acc, &reduction_params);
        assert_eq!(Uint::<4>::from_words(reduced), expected);
    }
}
