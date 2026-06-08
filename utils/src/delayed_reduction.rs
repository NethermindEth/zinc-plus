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
use std::marker::PhantomData;

const DEFAULT_DMR_FLUSH_ADDS: usize = 1 << 20;

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

/// Algorithm object for delayed modular reduction.
pub trait DelayedModularReductionAlgorithm {
    type Value;
    type Accumulator;

    fn zero_accumulator(&self) -> Self::Accumulator;
    fn add(&self, acc: &mut Self::Accumulator, value: &Self::Value);
    fn reduce(&self, acc: Self::Accumulator) -> Self::Value;
}

/// Algorithm object for delayed field product sums.
pub trait DelayedFieldProductSumAlgorithm {
    type Value;
    type Accumulator;

    fn zero_accumulator(&self) -> Self::Accumulator;
    fn add_product(&self, acc: &mut Self::Accumulator, lhs: &Self::Value, rhs: &Self::Value);
    fn reduce_products(&self, acc: Self::Accumulator) -> Self::Value;
    fn sum_of_products(&self, lhs: &[Self::Value], rhs: &[Self::Value]) -> Self::Value;
    fn sum_of_products_with_seed(
        &self,
        lhs: &[Self::Value],
        rhs: &[Self::Value],
        seed: Self::Value,
    ) -> Self::Value;
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

#[derive(Clone, Debug)]
pub struct BarrettDelayedReduction<'cfg, F>
where
    F: MontgomeryLimbs,
{
    cfg: &'cfg F::Config,
    params: BarrettReductionParams,
    flush_adds: usize,
    _field: PhantomData<F>,
}

impl<'cfg, F> BarrettDelayedReduction<'cfg, F>
where
    F: MontgomeryLimbs,
{
    pub fn new(cfg: &'cfg F::Config) -> Self {
        let params = F::barrett_reduction_params(cfg);
        let flush_adds = if params.modulus[3] == 0 {
            1
        } else {
            DEFAULT_DMR_FLUSH_ADDS
        };
        Self {
            cfg,
            params,
            flush_adds,
            _field: PhantomData,
        }
    }

    pub fn flush_adds(&self) -> usize {
        self.flush_adds
    }

    pub fn params(&self) -> &BarrettReductionParams {
        &self.params
    }
}

impl<F> DelayedModularReductionAlgorithm for BarrettDelayedReduction<'_, F>
where
    F: MontgomeryLimbs + Send + Sync,
{
    type Value = F;
    type Accumulator = Uint<5>;

    fn zero_accumulator(&self) -> Self::Accumulator {
        Uint::zero()
    }

    #[inline(always)]
    fn add(&self, acc: &mut Self::Accumulator, value: &Self::Value) {
        add_montgomery_limbs_5(acc, value.montgomery_limbs());
    }

    #[inline(always)]
    fn reduce(&self, acc: Self::Accumulator) -> Self::Value {
        F::from_montgomery_limbs(barrett_reduce_5(acc.as_words(), &self.params), self.cfg)
    }
}

/// Raw accumulator for a delayed sum of 4-limb Montgomery products.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProductAccumulator4 {
    limbs: Uint<9>,
    pending_products: usize,
}

impl ProductAccumulator4 {
    pub fn pending_products(&self) -> usize {
        self.pending_products
    }

    pub fn limbs(&self) -> &Uint<9> {
        &self.limbs
    }
}

#[derive(Clone, Debug)]
pub struct MontgomeryProductSum4<'cfg, F>
where
    F: MontgomeryLimbs,
{
    cfg: &'cfg F::Config,
    reduction_params: BarrettReductionParams,
    mod_neg_inv: u64,
    flush_products: usize,
    _field: PhantomData<F>,
}

impl<'cfg, F> MontgomeryProductSum4<'cfg, F>
where
    F: MontgomeryLimbs,
{
    pub fn new(cfg: &'cfg F::Config) -> Self {
        let reduction_params = F::barrett_reduction_params(cfg);
        let leading_zeros = clz::<4>(&reduction_params.modulus);
        let flush_products = if leading_zeros == 0 {
            1
        } else {
            usize::try_from(leading_zeros)
                .ok()
                .and_then(|shift| 1usize.checked_shl(shift as u32))
                .unwrap_or(usize::MAX)
        };
        Self::new_with_flush_products(cfg, flush_products)
    }

    pub fn new_with_flush_products(cfg: &'cfg F::Config, flush_products: usize) -> Self {
        let reduction_params = F::barrett_reduction_params(cfg);
        Self {
            cfg,
            reduction_params,
            mod_neg_inv: mod_neg_inv_u64(reduction_params.modulus[0]),
            flush_products: flush_products.max(1),
            _field: PhantomData,
        }
    }

    pub fn flush_products(&self) -> usize {
        self.flush_products
    }
}

impl<F> DelayedFieldProductSumAlgorithm for MontgomeryProductSum4<'_, F>
where
    F: MontgomeryLimbs + Send + Sync,
{
    type Value = F;
    type Accumulator = ProductAccumulator4;

    fn zero_accumulator(&self) -> Self::Accumulator {
        ProductAccumulator4 {
            limbs: Uint::zero(),
            pending_products: 0,
        }
    }

    #[inline(always)]
    fn add_product(&self, acc: &mut Self::Accumulator, lhs: &Self::Value, rhs: &Self::Value) {
        debug_assert!(
            acc.pending_products < self.flush_products,
            "ProductAccumulator4 must be reduced before exceeding its flush threshold"
        );
        add_montgomery_product_4x4(
            &mut acc.limbs,
            lhs.montgomery_limbs(),
            rhs.montgomery_limbs(),
        );
        acc.pending_products = acc.pending_products.saturating_add(1);
    }

    fn reduce_products(&self, acc: Self::Accumulator) -> Self::Value {
        if acc.pending_products == 0 {
            return F::zero_with_cfg(self.cfg);
        }
        let reduced = montgomery_reduce_9_to_4(
            acc.limbs.as_words(),
            &self.reduction_params,
            self.mod_neg_inv,
        );
        F::from_montgomery_limbs(reduced, self.cfg)
    }

    fn sum_of_products(&self, lhs: &[Self::Value], rhs: &[Self::Value]) -> Self::Value {
        let mut total = F::zero_with_cfg(self.cfg);
        let mut acc = self.zero_accumulator();
        for (left, right) in lhs.iter().zip(rhs) {
            self.add_product(&mut acc, left, right);
            if acc.pending_products >= self.flush_products {
                let pending = acc;
                total += self.reduce_products(pending);
                acc = self.zero_accumulator();
            }
        }
        if acc.pending_products != 0 {
            total += self.reduce_products(acc);
        }
        total
    }

    fn sum_of_products_with_seed(
        &self,
        lhs: &[Self::Value],
        rhs: &[Self::Value],
        seed: Self::Value,
    ) -> Self::Value {
        seed + self.sum_of_products(lhs, rhs)
    }
}

impl<F> DelayedModularReduction<F> for Uint<5>
where
    F: MontgomeryLimbs + Send + Sync,
{
    #[inline(always)]
    fn add(&mut self, value: &F) {
        add_montgomery_limbs_5(self, value.montgomery_limbs());
    }

    #[inline(always)]
    fn reduce(self, cfg: &F::Config, params: &BarrettReductionParams) -> F {
        let acc = self.as_words();
        F::from_montgomery_limbs(barrett_reduce_5(acc, params), cfg)
    }
}

#[inline(always)]
fn add_montgomery_limbs_5(acc: &mut Uint<5>, rhs: &[u64; 4]) {
    let acc = acc.as_mut_words();
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
fn mod_neg_inv_u64(modulus_limb: u64) -> u64 {
    debug_assert!(modulus_limb & 1 == 1, "Montgomery modulus must be odd");
    let mut inv = 1u64;
    let mut i = 0;
    while i < 6 {
        inv = inv.wrapping_mul(2u64.wrapping_sub(modulus_limb.wrapping_mul(inv)));
        i += 1;
    }
    inv.wrapping_neg()
}

#[inline(always)]
fn add_montgomery_product_4x4(acc: &mut Uint<9>, lhs: &[u64; 4], rhs: &[u64; 4]) {
    let product = mul_4x4_to_8(lhs, rhs);
    let acc = acc.as_mut_words();
    let mut carry = 0u64;
    let mut i = 0;
    while i < 8 {
        let (sum, c0) = acc[i].overflowing_add(product[i]);
        let (sum, c1) = sum.overflowing_add(carry);
        acc[i] = sum;
        carry = (c0 as u64) + (c1 as u64);
        i += 1;
    }

    let old_hi = acc[8];
    acc[8] = acc[8].wrapping_add(carry);
    debug_assert!(acc[8] >= old_hi, "ProductAccumulator4 overflowed high limb");
}

#[inline(always)]
fn mul_4x4_to_8(lhs: &[u64; 4], rhs: &[u64; 4]) -> [u64; 8] {
    let mut result = [0u64; 8];
    let mut i = 0;
    while i < 4 {
        let mut carry = 0u128;
        let mut j = 0;
        while j < 4 {
            let idx = i + j;
            let prod = (lhs[i] as u128) * (rhs[j] as u128) + (result[idx] as u128) + carry;
            result[idx] = prod as u64;
            carry = prod >> 64;
            j += 1;
        }

        let mut idx = i + 4;
        let mut carry_u64 = carry as u64;
        while carry_u64 != 0 && idx < 8 {
            let (sum, overflow) = result[idx].overflowing_add(carry_u64);
            result[idx] = sum;
            carry_u64 = overflow as u64;
            idx += 1;
        }
        debug_assert!(carry_u64 == 0, "4x4 product exceeded eight limbs");
        i += 1;
    }
    result
}

#[inline(always)]
fn montgomery_reduce_9_to_4(
    acc: &[u64; 9],
    params: &BarrettReductionParams,
    mod_neg_inv: u64,
) -> [u64; 4] {
    let mut t = *acc;
    let mut i = 0;
    while i < 4 {
        let q = t[i].wrapping_mul(mod_neg_inv);
        let mut carry = 0u128;
        let mut j = 0;
        while j < 4 {
            let idx = i + j;
            let sum = (q as u128) * (params.modulus[j] as u128) + (t[idx] as u128) + carry;
            t[idx] = sum as u64;
            carry = sum >> 64;
            j += 1;
        }

        let mut idx = i + 4;
        let mut carry_u64 = carry as u64;
        while carry_u64 != 0 {
            debug_assert!(idx < 9, "Montgomery reduction carry exceeded accumulator");
            let (sum, overflow) = t[idx].overflowing_add(carry_u64);
            t[idx] = sum;
            carry_u64 = overflow as u64;
            idx += 1;
        }
        debug_assert!(t[i] == 0, "Montgomery reduction did not clear low limb");
        i += 1;
    }

    let reduced = [t[4], t[5], t[6], t[7], t[8]];
    barrett_reduce_5(&reduced, params)
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

    fn batched_product_cfg() -> <F as PrimeField>::Config {
        let modulus = Uint::new(
            crypto_bigint::Uint::<4>::from_str_radix_vartime(
                "00dca94d8a1ecce3b6e8755d8999787d0524d8ca1ea755e7af84fb646fa31f27",
                16,
            )
            .expect("valid modulus"),
        );
        F::make_cfg(&modulus).expect("valid field config")
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
        let reducer = BarrettDelayedReduction::<F>::new(&cfg);
        let values: Vec<F> = (0..512)
            .map(|i| F::from_with_cfg(i as u64 + 1, &cfg))
            .collect();

        let mut expected = F::zero_with_cfg(&cfg);
        for value in &values {
            expected += value;
        }

        let mut acc = reducer.zero_accumulator();
        for value in &values {
            reducer.add(&mut acc, value);
        }

        assert_eq!(reducer.reduce(acc), expected);
    }

    #[test]
    fn barrett_reduce_5_matches_uint_remainder_for_bounded_sum() {
        let cfg = secp256k1_cfg();
        let reduction_params = F::barrett_reduction_params(&cfg);
        let reducer = BarrettDelayedReduction::<F>::new(&cfg);
        let mut acc = Uint::<5>::zero();
        let max = -F::from_with_cfg(1u64, &cfg);
        for _ in 0..512 {
            reducer.add(&mut acc, &max);
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

    #[test]
    fn product_accumulator_single_product_matches_field_multiplication() {
        let cfg = secp256k1_cfg();
        let reducer = MontgomeryProductSum4::<F>::new(&cfg);
        let lhs = F::from_with_cfg(17u64, &cfg);
        let rhs = F::from_with_cfg(23u64, &cfg);

        let mut acc = reducer.zero_accumulator();
        reducer.add_product(&mut acc, &lhs, &rhs);

        assert_eq!(reducer.reduce_products(acc), lhs * &rhs);
    }

    #[test]
    fn product_accumulator_multi_product_matches_naive_sum() {
        let cfg = secp256k1_cfg();
        let reducer = MontgomeryProductSum4::<F>::new(&cfg);
        let lhs: Vec<F> = (0..32).map(|idx| F::from_with_cfg(idx + 3, &cfg)).collect();
        let rhs: Vec<F> = (0..32)
            .map(|idx| F::from_with_cfg(257 - idx, &cfg))
            .collect();

        let expected = lhs
            .iter()
            .zip(&rhs)
            .fold(F::zero_with_cfg(&cfg), |acc, (left, right)| {
                acc + left.clone() * right
            });

        assert_eq!(reducer.sum_of_products(&lhs, &rhs), expected);
    }

    #[test]
    fn product_accumulator_batches_near_modulus_terms_before_reduction() {
        let cfg = batched_product_cfg();
        let reducer = MontgomeryProductSum4::<F>::new(&cfg);
        assert!(reducer.flush_products() > 64);

        let lhs: Vec<F> = (0..64)
            .map(|idx| -F::from_with_cfg(idx * 17 + 5, &cfg))
            .collect();
        let rhs: Vec<F> = (0..64)
            .map(|idx| -F::from_with_cfg(idx * 19 + 7, &cfg))
            .collect();

        let mut acc = reducer.zero_accumulator();
        for (left, right) in lhs.iter().zip(&rhs) {
            reducer.add_product(&mut acc, left, right);
        }
        assert_eq!(acc.pending_products(), lhs.len());

        let expected = lhs
            .iter()
            .zip(&rhs)
            .fold(F::zero_with_cfg(&cfg), |sum, (left, right)| {
                sum + left.clone() * right
            });

        assert_eq!(reducer.reduce_products(acc), expected);
    }

    #[test]
    fn product_accumulator_seeded_sum_matches_naive_sum() {
        let cfg = secp256k1_cfg();
        let reducer = MontgomeryProductSum4::<F>::new(&cfg);
        let seed = F::from_with_cfg(99u64, &cfg);
        let lhs: Vec<F> = (0..16).map(|idx| F::from_with_cfg(idx + 5, &cfg)).collect();
        let rhs: Vec<F> = (0..16)
            .map(|idx| F::from_with_cfg(131 - idx, &cfg))
            .collect();

        let expected = lhs
            .iter()
            .zip(&rhs)
            .fold(seed.clone(), |acc, (left, right)| {
                acc + left.clone() * right
            });

        assert_eq!(
            reducer.sum_of_products_with_seed(&lhs, &rhs, seed),
            expected
        );
    }

    #[test]
    fn product_accumulator_forced_flush_matches_naive_sum() {
        let cfg = secp256k1_cfg();
        let reducer = MontgomeryProductSum4::<F>::new_with_flush_products(&cfg, 1);
        let lhs: Vec<F> = (0..32)
            .map(|idx| F::from_with_cfg(idx + 11, &cfg))
            .collect();
        let rhs: Vec<F> = (0..32)
            .map(|idx| F::from_with_cfg(409 - idx, &cfg))
            .collect();

        let expected = lhs
            .iter()
            .zip(&rhs)
            .fold(F::zero_with_cfg(&cfg), |acc, (left, right)| {
                acc + left.clone() * right
            });

        assert_eq!(reducer.sum_of_products(&lhs, &rhs), expected);
    }
}
