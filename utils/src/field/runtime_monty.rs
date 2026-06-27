//! A value-sized prime field with a **runtime-installed** modulus.
//!
//! # Motivation
//!
//! On the `main-beta` integer prover path the projected/"random" field is
//! [`crypto_primitives::crypto_bigint_monty::MontyField<LIMBS>`], a crypto-bigint
//! Montgomery field whose modulus is chosen at runtime (drawn from the
//! Fiat–Shamir transcript, or fixed to the secp256k1 base prime).
//!
//! Because the modulus is a *runtime* value, that type wraps
//! `crypto_bigint::modular::MontyForm`, which stores the full
//! [`MontyParams`] (modulus, `R`, `R²`, `mod_inv`, …) **inline, in every single
//! element**. For a 256-bit prime (`LIMBS = 4`):
//!
//! | type                       | bytes / element | overhead |
//! |----------------------------|-----------------|----------|
//! | `MontyForm<4>`             | ~144            | 4.5×     |
//! | `ConstMontyForm<_, 4>`     | 32              | 1×       |
//! | [`Fp<_, 4>`] (this module) | 32              | 1×       |
//!
//! The kicker: there is exactly **one** modulus per proof — it is built once
//! (e.g. `secp256k1_field_cfg`) and then *cloned into every one of the millions
//! of field elements*. Storing per-element what is in fact a single ambient
//! constant is the waste this module removes.
//!
//! # Approach
//!
//! `ConstMontyForm` already gives value-sized elements — but it needs the
//! modulus as a compile-time `const` (a `ConstMontyParams` type), which a
//! transcript-drawn prime cannot be.
//!
//! [`Fp`] threads the needle: the modulus is a runtime value, but it is stored
//! **once** in a process-global [`OnceLock`] selected by a zero-sized *slot*
//! type `S` (see [`Modulus`] / [`define_modulus!`]). The element itself is just
//! the Montgomery-form `Uint<LIMBS>` — identical footprint to `ConstMontyForm`.
//! Arithmetic reads the shared params through the slot (a lock-free atomic load
//! after one-time install), so no element ever carries the config.
//!
//! The slot caches the Montgomery `mod_neg_inv` alongside the params, so the hot
//! multiply (a faithful CIOS port — see [`mont_mul`]) reads only references and
//! does no per-op recomputation: it matches `MontyForm`'s arithmetic cost while
//! moving 4.5× fewer bytes.
//!
//! This keeps ordinary operator ergonomics (`a * b`, `a + b`) — unlike the
//! partial "store `F::Inner` and thread `&cfg` everywhere" workaround currently
//! scattered across `poly`/`piop` — while being a drop-in for the value layout.
//!
//! # Constraints / tradeoffs
//!
//! * **One modulus per slot per process.** A slot is set once; re-installing the
//!   *same* modulus is a no-op, re-installing a *different* one panics. The
//!   `main-beta` prover uses a single fixed prime, so one slot suffices. Code
//!   that must juggle several moduli in one process declares several slots.
//! * The modulus must be installed (via [`Modulus::install`] /
//!   [`Modulus::install_modulus`]) before any arithmetic; otherwise the ambient
//!   lookup panics with a clear message.
//!
//! See `documentation/runtime-const-field-design.md` for the full rationale,
//! measurements, and the migration path to make this a drop-in `F`.

// Field operators (`+`, `-`, `*` on `Fp`) are total, so `arithmetic_side_effects`
// is a false positive for them — the same "False alert" convention the crate's
// existing `field/monty.rs` uses. The only genuinely overflow-prone integer
// arithmetic is the audited CIOS port in `mont_mul`.
#![allow(clippy::arithmetic_side_effects)]

use core::fmt;
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;
use std::sync::OnceLock;

use crypto_bigint::modular::{MontyForm, MontyParams};
use crypto_bigint::{NonZero, Odd, Uint, Word};

/// The ambient parameters for a runtime modulus: the crypto-bigint
/// [`MontyParams`] plus the cached Montgomery `mod_neg_inv` (so the hot multiply
/// never recomputes it). One instance lives per slot — never per element.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Params<const LIMBS: usize> {
    monty: MontyParams<LIMBS>,
    mod_neg_inv: Word,
}

impl<const LIMBS: usize> Params<LIMBS> {
    /// Build from a `MontyParams`, caching `mod_neg_inv`.
    pub fn new(monty: MontyParams<LIMBS>) -> Self {
        let mod_neg_inv = mont_mul::compute_mod_neg_inv(monty.modulus().as_ref().as_words()[0]);
        Self { monty, mod_neg_inv }
    }

    #[inline(always)]
    fn modulus(&self) -> &Uint<LIMBS> {
        self.monty.modulus().as_ref()
    }

    #[inline(always)]
    fn modulus_nz(&self) -> &NonZero<Uint<LIMBS>> {
        self.monty.modulus().as_nz_ref()
    }
}

/// A compile-time tag selecting a single runtime-installed modulus.
///
/// Each implementing (zero-sized) type owns exactly one `OnceLock<Params>` via
/// [`Modulus::cell`]. Use [`define_modulus!`] to declare one.
pub trait Modulus<const LIMBS: usize>:
    Copy + Clone + PartialEq + Eq + fmt::Debug + Send + Sync + 'static
{
    /// The process-global cell holding this slot's parameters.
    fn cell() -> &'static OnceLock<Params<LIMBS>>;

    /// The installed parameters. Panics if the modulus was never installed.
    #[inline(always)]
    fn params() -> &'static Params<LIMBS> {
        Self::cell()
            .get()
            .expect("runtime_monty: modulus not installed for this slot; call install() first")
    }

    /// Install ready-made [`Params`]. Idempotent for the *same* modulus; panics
    /// if a *different* modulus was already installed in this slot.
    fn install(params: Params<LIMBS>) {
        if let Err(_already_set) = Self::cell().set(params) {
            assert!(
                Self::params() == &params,
                "runtime_monty: slot already installed with a different modulus"
            );
        }
    }

    /// Install from crypto-bigint [`MontyParams`] (e.g. an existing `F::Config`).
    fn install_monty(monty: MontyParams<LIMBS>) {
        Self::install(Params::new(monty));
    }

    /// Install from a raw modulus value (must be odd, as Montgomery requires).
    fn install_modulus(modulus: Uint<LIMBS>) {
        let odd = Odd::new(modulus)
            .into_option()
            .expect("runtime_monty: modulus must be odd");
        Self::install(Params::new(MontyParams::new(odd)));
    }
}

/// Declare a zero-sized modulus slot implementing [`Modulus`].
///
/// ```ignore
/// define_modulus!(pub Secp256k1Base, 4);
/// Secp256k1Base::install_modulus(p);     // once, at proof start
/// type F = Fp<Secp256k1Base, 4>;
/// ```
#[macro_export]
macro_rules! define_modulus {
    ($vis:vis $name:ident, $limbs:expr) => {
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        $vis struct $name;

        impl $crate::field::runtime_monty::Modulus<{ $limbs }> for $name {
            #[inline(always)]
            fn cell() -> &'static ::std::sync::OnceLock<
                $crate::field::runtime_monty::Params<{ $limbs }>,
            > {
                static CELL: ::std::sync::OnceLock<
                    $crate::field::runtime_monty::Params<{ $limbs }>,
                > = ::std::sync::OnceLock::new();
                &CELL
            }
        }
    };
}

/// A prime-field element with a runtime-installed modulus, stored in Montgomery
/// form as a bare `Uint<LIMBS>` (no per-element config).
///
/// The modulus is taken from slot `S`'s ambient [`Params`].
pub struct Fp<S: Modulus<LIMBS>, const LIMBS: usize> {
    /// The value in Montgomery form (`x · R mod m`), matching `MontyForm`'s
    /// internal representation.
    mont: Uint<LIMBS>,
    _slot: PhantomData<S>,
}

impl<S: Modulus<LIMBS>, const LIMBS: usize> Fp<S, LIMBS> {
    /// Wrap a value already in Montgomery form.
    #[inline(always)]
    pub const fn from_montgomery(mont: Uint<LIMBS>) -> Self {
        Self {
            mont,
            _slot: PhantomData,
        }
    }

    /// The raw Montgomery-form limbs (what [`Self::from_montgomery`] consumes).
    #[inline(always)]
    pub const fn to_montgomery(&self) -> Uint<LIMBS> {
        self.mont
    }

    /// Build from a canonical (non-Montgomery) integer, reducing mod the
    /// installed modulus. `mont = value · R² · R⁻¹ = value · R`.
    #[inline]
    pub fn new(value: Uint<LIMBS>) -> Self {
        let p = S::params();
        Self::from_montgomery(mont_mul::monty_mul(
            &value,
            p.monty.r2(),
            p.modulus(),
            p.mod_neg_inv,
        ))
    }

    /// Build from a `u64`.
    #[inline]
    pub fn from_u64(value: u64) -> Self {
        Self::new(Uint::from_u64(value))
    }

    /// The canonical (reduced, non-Montgomery) representative.
    /// `value = mont · 1 · R⁻¹`.
    #[inline]
    pub fn retrieve(&self) -> Uint<LIMBS> {
        let p = S::params();
        mont_mul::monty_mul(&self.mont, &Uint::ONE, p.modulus(), p.mod_neg_inv)
    }

    /// Additive identity (`0` is `0` in Montgomery form too).
    #[inline(always)]
    pub const fn zero() -> Self {
        Self::from_montgomery(Uint::ZERO)
    }

    /// Multiplicative identity (`1 · R mod m`, cached in params).
    #[inline]
    pub fn one() -> Self {
        Self::from_montgomery(*S::params().monty.one())
    }

    /// True iff this is the additive identity.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.mont == Uint::ZERO
    }

    /// Multiplicative inverse, or `None` for zero. (Not a hot path; routed
    /// through `MontyForm` for a vetted constant-time inversion.)
    #[inline]
    pub fn inv(&self) -> Option<Self> {
        let form = MontyForm::from_montgomery(self.mont, S::params().monty);
        Option::from(form.invert())
            .map(|i: MontyForm<LIMBS>| Self::from_montgomery(*i.as_montgomery()))
    }

    /// Exponentiation by a small exponent (square-and-multiply).
    #[inline]
    pub fn pow(&self, mut exp: u32) -> Self {
        let mut base = *self;
        let mut acc = Self::one();
        while exp > 0 {
            if exp & 1 == 1 {
                acc *= base;
            }
            base *= base;
            exp >>= 1;
        }
        acc
    }

    #[inline(always)]
    fn mul_mont(&self, rhs: &Self) -> Uint<LIMBS> {
        let p = S::params();
        mont_mul::monty_mul(&self.mont, &rhs.mont, p.modulus(), p.mod_neg_inv)
    }
}

// --- lightweight trait impls touching only `mont` (no spurious S bounds) ---

impl<S: Modulus<LIMBS>, const LIMBS: usize> Clone for Fp<S, LIMBS> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}
impl<S: Modulus<LIMBS>, const LIMBS: usize> Copy for Fp<S, LIMBS> {}

impl<S: Modulus<LIMBS>, const LIMBS: usize> PartialEq for Fp<S, LIMBS> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.mont == other.mont
    }
}
impl<S: Modulus<LIMBS>, const LIMBS: usize> Eq for Fp<S, LIMBS> {}

impl<S: Modulus<LIMBS>, const LIMBS: usize> Hash for Fp<S, LIMBS> {
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.mont.hash(state);
    }
}

impl<S: Modulus<LIMBS>, const LIMBS: usize> Default for Fp<S, LIMBS> {
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

impl<S: Modulus<LIMBS>, const LIMBS: usize> fmt::Debug for Fp<S, LIMBS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fp({})", self.retrieve())
    }
}

impl<S: Modulus<LIMBS>, const LIMBS: usize> fmt::Display for Fp<S, LIMBS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.retrieve())
    }
}

// --- arithmetic operators ---

use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

impl<S: Modulus<LIMBS>, const LIMBS: usize> Add for Fp<S, LIMBS> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        // Montgomery form is additively linear, so add_mod on raw limbs is exact.
        Self::from_montgomery(self.mont.add_mod(&rhs.mont, S::params().modulus_nz()))
    }
}

impl<S: Modulus<LIMBS>, const LIMBS: usize> Sub for Fp<S, LIMBS> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::from_montgomery(self.mont.sub_mod(&rhs.mont, S::params().modulus_nz()))
    }
}

impl<S: Modulus<LIMBS>, const LIMBS: usize> Mul for Fp<S, LIMBS> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self::from_montgomery(self.mul_mont(&rhs))
    }
}

impl<S: Modulus<LIMBS>, const LIMBS: usize> Neg for Fp<S, LIMBS> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::from_montgomery(self.mont.neg_mod(S::params().modulus_nz()))
    }
}

// reference-rhs variants (match the `Semiring` trait surface)
impl<S: Modulus<LIMBS>, const LIMBS: usize> Add<&Fp<S, LIMBS>> for Fp<S, LIMBS> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: &Self) -> Self {
        self + *rhs
    }
}
impl<S: Modulus<LIMBS>, const LIMBS: usize> Sub<&Fp<S, LIMBS>> for Fp<S, LIMBS> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: &Self) -> Self {
        self - *rhs
    }
}
impl<S: Modulus<LIMBS>, const LIMBS: usize> Mul<&Fp<S, LIMBS>> for Fp<S, LIMBS> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: &Self) -> Self {
        self * *rhs
    }
}

impl<S: Modulus<LIMBS>, const LIMBS: usize> AddAssign for Fp<S, LIMBS> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl<S: Modulus<LIMBS>, const LIMBS: usize> SubAssign for Fp<S, LIMBS> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl<S: Modulus<LIMBS>, const LIMBS: usize> MulAssign for Fp<S, LIMBS> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl<S: Modulus<LIMBS>, const LIMBS: usize> AddAssign<&Fp<S, LIMBS>> for Fp<S, LIMBS> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: &Self) {
        *self = *self + *rhs;
    }
}
impl<S: Modulus<LIMBS>, const LIMBS: usize> SubAssign<&Fp<S, LIMBS>> for Fp<S, LIMBS> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &Self) {
        *self = *self - *rhs;
    }
}
impl<S: Modulus<LIMBS>, const LIMBS: usize> MulAssign<&Fp<S, LIMBS>> for Fp<S, LIMBS> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &Self) {
        *self = *self * *rhs;
    }
}

/// CIOS Montgomery multiplication on bare limbs — a faithful port of the
/// `pub(crate)` helper in `crypto_primitives::field::crypto_bigint_helpers`
/// (which we cannot call across the crate boundary). Verified element-for-element
/// against `MontyForm` in this module's property tests.
pub(crate) mod mont_mul {
    #![allow(clippy::arithmetic_side_effects, clippy::cast_possible_truncation)]
    use crypto_bigint::{Uint, WideWord, Word};
    use num_traits::ConstZero;

    const LOG2_WORD_BITS: u32 = Word::BITS.trailing_zeros();

    /// `-modulus⁻¹ mod 2^word_bits` via Newton's method (6 iters for 64-bit).
    #[inline(always)]
    pub const fn compute_mod_neg_inv(m0: Word) -> Word {
        const TWO: Word = 2;
        let mut inv: Word = 1;
        let mut i = 0;
        while i < LOG2_WORD_BITS {
            inv = inv.wrapping_mul(TWO.wrapping_sub(m0.wrapping_mul(inv)));
            i += 1;
        }
        inv.wrapping_neg()
    }

    /// `a · b · R⁻¹ mod m` for limbs already in Montgomery form, with a
    /// precomputed `mod_neg_inv` (no per-call Newton iteration).
    #[inline]
    pub fn monty_mul<const LIMBS: usize>(
        a: &Uint<LIMBS>,
        b: &Uint<LIMBS>,
        modulus: &Uint<LIMBS>,
        mod_neg_inv: Word,
    ) -> Uint<LIMBS> {
        let a_words = a.as_words();
        let b_words = b.as_words();
        let mod_words = modulus.as_words();

        let mut result = [0; LIMBS];
        let carry =
            montgomery_mul_cios::<LIMBS>(a_words, b_words, mod_words, mod_neg_inv, &mut result);

        // Conditional subtraction: use (result - modulus) iff result overflowed
        // (carry != 0) or result >= modulus (no borrow).
        let mut diff = [0; LIMBS];
        let mut borrow: Word = 0;
        for i in 0..LIMBS {
            let (d, b1) = result[i].overflowing_sub(mod_words[i]);
            let (d, b2) = d.overflowing_sub(borrow);
            diff[i] = d;
            borrow = Word::from(b1) | Word::from(b2);
        }
        let use_diff = (carry != 0) | (borrow == 0);
        let mask = Word::ZERO.wrapping_sub(Word::from(use_diff));
        for i in 0..LIMBS {
            result[i] = (diff[i] & mask) | (result[i] & !mask);
        }
        Uint::from_words(result)
    }

    #[inline]
    fn montgomery_mul_cios<const LIMBS: usize>(
        a: &[Word; LIMBS],
        b: &[Word; LIMBS],
        modulus: &[Word; LIMBS],
        mod_neg_inv: Word,
        out: &mut [Word; LIMBS],
    ) -> Word {
        let mut acc_hi: Word = 0;
        for &a_i in a {
            let mut carry = 0;
            for j in 0..LIMBS {
                let (lo, hi) = mul_add_carry(a_i, b[j], out[j], carry);
                out[j] = lo;
                carry = hi;
            }
            let (new_acc_hi, meta_carry) = acc_hi.overflowing_add(carry);
            acc_hi = new_acc_hi;

            let u = out[0].wrapping_mul(mod_neg_inv);
            let (_, hi) = mul_add_carry(u, modulus[0], out[0], 0);
            carry = hi;
            for j in 1..LIMBS {
                let (lo, hi) = mul_add_carry(u, modulus[j], out[j], carry);
                out[j - 1] = lo;
                carry = hi;
            }
            let (sum, c) = acc_hi.overflowing_add(carry);
            out[LIMBS - 1] = sum;
            acc_hi = Word::from(meta_carry) + Word::from(c);
        }
        acc_hi
    }

    /// `a · b + c + d` returning `(lo, hi)`.
    #[inline(always)]
    fn mul_add_carry(a: Word, b: Word, c: Word, d: Word) -> (Word, Word) {
        let wide = WideWord::from(a) * WideWord::from(b) + WideWord::from(c) + WideWord::from(d);
        (wide as Word, (wide >> Word::BITS) as Word)
    }
}

#[cfg(test)]
#[allow(
    clippy::arithmetic_side_effects,
    clippy::unwrap_used,
    clippy::cast_possible_truncation
)]
mod tests {
    use super::*;
    use crypto_bigint::U256;
    use proptest::prelude::*;

    // A 256-bit prime (the modulus used in the existing crypto_primitives field
    // property tests), installed into a dedicated slot for this test module.
    const MODULUS_HEX: &str =
        "00dca94d8a1ecce3b6e8755d8999787d0524d8ca1ea755e7af84fb646fa31f27";

    define_modulus!(TestMod, 4);

    type F = Fp<TestMod, 4>;

    fn modulus() -> U256 {
        U256::from_be_hex(MODULUS_HEX)
    }

    fn ref_params() -> MontyParams<4> {
        MontyParams::new(Odd::new(modulus()).into_option().unwrap())
    }

    fn setup() {
        TestMod::install_modulus(modulus());
    }

    // Reference element: crypto-bigint's MontyForm (what `MontyField` wraps).
    fn refr(x: U256) -> MontyForm<4> {
        MontyForm::new(&x, ref_params())
    }

    fn any_u256() -> impl Strategy<Value = U256> {
        any::<[u8; 32]>().prop_map(|b| U256::from_le_slice(&b))
    }

    #[test]
    fn element_is_value_sized() {
        // The whole point: an element is exactly the Uint, with the slot a ZST.
        assert_eq!(core::mem::size_of::<F>(), core::mem::size_of::<U256>());
        // ... and dramatically smaller than the config-carrying MontyForm.
        assert!(core::mem::size_of::<F>() < core::mem::size_of::<MontyForm<4>>());
    }

    #[test]
    fn zero_one_roundtrip() {
        setup();
        assert_eq!(F::zero().retrieve(), U256::ZERO);
        assert_eq!(F::one().retrieve(), U256::ONE);
        assert!(F::zero().is_zero());
        assert!(!F::one().is_zero());
    }

    #[test]
    fn from_u64_and_new_agree() {
        setup();
        assert_eq!(F::from_u64(12345).retrieve(), U256::from_u64(12345));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(400))]

        #[test]
        fn add_matches_reference(a in any_u256(), b in any_u256()) {
            setup();
            let got = (F::new(a) + F::new(b)).retrieve();
            let want = (refr(a) + refr(b)).retrieve();
            prop_assert_eq!(got, want);
        }

        #[test]
        fn sub_matches_reference(a in any_u256(), b in any_u256()) {
            setup();
            let got = (F::new(a) - F::new(b)).retrieve();
            let want = (refr(a) - refr(b)).retrieve();
            prop_assert_eq!(got, want);
        }

        #[test]
        fn mul_matches_reference(a in any_u256(), b in any_u256()) {
            setup();
            let got = (F::new(a) * F::new(b)).retrieve();
            let want = (refr(a) * refr(b)).retrieve();
            prop_assert_eq!(got, want);
        }

        #[test]
        fn neg_matches_reference(a in any_u256()) {
            setup();
            let got = (-F::new(a)).retrieve();
            let want = (-refr(a)).retrieve();
            prop_assert_eq!(got, want);
        }

        #[test]
        fn roundtrip_new_retrieve(a in any_u256()) {
            setup();
            // new() reduces mod m, so compare against the reference reduction.
            prop_assert_eq!(F::new(a).retrieve(), refr(a).retrieve());
        }

        #[test]
        fn inv_is_multiplicative_identity(a in any_u256()) {
            setup();
            let f = F::new(a);
            if !f.is_zero() {
                let inv = f.inv().unwrap();
                prop_assert_eq!((f * inv).retrieve(), U256::ONE);
            }
        }

        #[test]
        fn pow_matches_repeated_mul(a in any_u256(), e in 0u32..64) {
            setup();
            let f = F::new(a);
            let mut want = F::one();
            for _ in 0..e {
                want = want * f;
            }
            prop_assert_eq!(f.pow(e).retrieve(), want.retrieve());
        }

        #[test]
        fn assign_ops_match(a in any_u256(), b in any_u256()) {
            setup();
            let (fa, fb) = (F::new(a), F::new(b));
            let mut s = fa; s += fb;
            let mut d = fa; d -= fb;
            let mut m = fa; m *= fb;
            prop_assert_eq!(s.retrieve(), (fa + fb).retrieve());
            prop_assert_eq!(d.retrieve(), (fa - fb).retrieve());
            prop_assert_eq!(m.retrieve(), (fa * fb).retrieve());
        }
    }
}
