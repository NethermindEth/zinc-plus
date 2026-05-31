//! `GF(2^128)`: degree-128 binary extension field.
//!
//! Elements are `F_2[X] / <f(X)>` where `f(X)` is the GHASH / AES-GCM
//! reduction polynomial
//!
//! ```text
//! f(X) = X^128 + X^7 + X^2 + X + 1.
//! ```
//!
//! `f` is irreducible over `F_2`. The factor ring is therefore a field
//! of order `2^128`. Its multiplicative group has order `2^128 - 1`,
//! so for every nonzero `a` we have `a^{2^128 - 1} = 1` and
//! `a^{-1} = a^{2^128 - 2}`.
//!
//! Storage layout: each element is a [`Uint<2>`] (2 × `u64` = 128 bits)
//! bit-packed polynomial of degree `< 128`. Bit `64*w + b` (LSB-first
//! within each `u64` limb) holds the coefficient of `X^{64*w + b}`.
//! Addition is XOR; multiplication is the F_2 carryless product
//! followed by reduction modulo `f`.
//!
//! Intended use: the random "projecting element" `α` in the F_2 proving
//! path. After the ideal check runs over `F_2[X]`, the protocol samples
//! `α ∈ GF(2^128)` and substitutes `X = α` in every committed cell, so
//! that the sumcheck-based phase runs over `GF(2^128)` instead of a
//! prime field. The 128-bit choice (vs the prior 192-bit field) cuts
//! the inner Karatsuba mul-count from 6 to 3 PMULL/PCLMUL ops per
//! field multiplication and the canonical-representative width from
//! `Uint<3>` to `Uint<2>`.

use core::{
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    hash::Hash,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};
use crypto_primitives::{
    Field, FieldError, PrimeField, Ring, Semiring, crypto_bigint_uint::Uint,
};
use zinc_utils::inner_transparent_field::InnerTransparentField;
use num_traits::{
    CheckedAdd, CheckedMul, CheckedNeg, CheckedSub, ConstOne, ConstZero, Inv, One, Pow, Zero,
};

use crate::univariate::{
    F2PackU64, binary::BinaryPoly, binary_f2_wide::BinaryF2Poly, binary_u64::BinaryU64Poly,
    dense::DensePolynomial,
};

/// A `GF(2^128)[X]<D>` polynomial — degree-`<D` univariate with
/// `GF(2^128)`-valued coefficients. The natural target of the
/// `F_2[X] → GF(2^128)[X]` coefficient lift used in step 2 of the
/// F_2 proving path (see `protocol/src/f2_prove_plan.md`).
///
/// Built on top of the existing [`DensePolynomial`] machinery:
/// `BinaryFieldGF128` already implements `Semiring` (via the
/// degenerate `PrimeField` impl), so addition / negation /
/// `EvaluatablePolynomial<R, R>` (Horner at a point) all come
/// for free.
pub type GF128Poly<const D: usize> = DensePolynomial<BinaryFieldGF128, D>;

/// Low bits of the reduction polynomial — `g(X) = X^7 + X^2 + X + 1`,
/// stored as `0x87` (bits 0, 1, 2, 7 set). The `X^128` term is implicit
/// in the reduction routine: `X^128 ≡ g(X) mod f`.
pub const REDUCTION_LOW_GF128: u64 = 0x87;

/// `Field::Modulus`-shaped representation of the reduction polynomial:
/// the same low-bit pattern as [`REDUCTION_LOW_GF128`], promoted to a
/// `Uint<2>` so the byte width matches `Field::Inner` (a requirement
/// of `Transcript::absorb_random_field`).
pub const MODULUS_LOW_BITS_GF128: Uint<2> = Uint::<2>::from_words([REDUCTION_LOW_GF128, 0]);

/// An element of `GF(2^128) = F_2[X] / <X^128 + X^7 + X^2 + X + 1>`.
///
/// Stored as a [`Uint<2>`] (2 × `u64` = 128 bits). Bit `64*w + b`
/// (LSB-first within each `u64` limb) holds the coefficient of
/// `X^{64*w + b}`; bits `0..128` carry the value, no bits at or
/// above 128 are ever set in a reduced element.
///
/// The choice of `Uint<2>` for storage is load-bearing for the
/// transcript / `Field::Inner` plumbing: `Uint<L>` already
/// implements `ConstTranscribable`, which is what the IC's
/// `transcript.get_field_challenge::<F>` API requires of
/// `F::Inner`.
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
#[repr(transparent)]
pub struct BinaryFieldGF128 {
    uint: Uint<2>,
}

impl BinaryFieldGF128 {
    #[inline]
    pub const fn zero() -> Self {
        Self { uint: Uint::<2>::ZERO }
    }

    #[inline]
    pub const fn one() -> Self {
        Self {
            uint: Uint::<2>::from_words([1u64, 0]),
        }
    }

    /// Construct from raw bit-packed words. Caller is responsible for
    /// ensuring the value is already reduced (no bits ≥ 128 set; here
    /// that's the natural invariant since the array only has 128 bits).
    #[inline]
    pub const fn from_words(words: [u64; 2]) -> Self {
        Self {
            uint: Uint::<2>::from_words(words),
        }
    }

    /// Borrow the bit-packed representation. Bit `i` of word `i / 64`
    /// holds the coefficient of `X^i`.
    #[inline]
    pub const fn words(&self) -> &[u64; 2] {
        self.uint.as_words()
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        let w = self.uint.as_words();
        w[0] == 0 && w[1] == 0
    }

    /// `a²` via carryless square + reduction.
    ///
    /// Squaring in characteristic 2 is the bit-expansion permutation
    /// (`bit i` → `bit 2i`). We could specialise this to a fixed
    /// PSHUFB/TBL lookup, but the generic `clmul_128x128(self, self)`
    /// path is plenty fast given the 3-inner-clmul Karatsuba layout.
    #[inline]
    pub fn square(&self) -> Self {
        let prod = clmul_128x128(self.uint.as_words(), self.uint.as_words());
        Self::from_words(reduce_256_to_128(prod))
    }

    /// `a^{-1}` via Fermat: `a^{-1} = a^{2^128 - 2}` for `a ≠ 0`.
    ///
    /// `2^128 - 2 = 2 · (2^127 - 1)`, so we compute `b = a^{2^127 - 1}`
    /// and return `b^2`. `b` is built via the standard "all-ones
    /// exponent" loop: if `c_k = a^{2^k - 1}` then
    /// `c_{k+1} = c_k^2 · a`. After 126 such steps from `c_1 = a`,
    /// `c_127 = a^{2^127 - 1}`. Total cost: 127 squarings + 126 mults.
    ///
    /// # TODO — Itoh–Tsujii addition chain (deferred optimisation)
    ///
    /// For `GF(2^n)`, Itoh–Tsujii computes `a^{-1} = (a^{2^n - 2}) =
    /// (a^r)^{2}` where `r = (2^{n-1} - 1)`, building `a^r` via the
    /// recurrence `c_{k+m} = (c_k)^{2^m} · c_m`. The total cost is
    /// `O(log n) mults + (n - 1) squarings` — for `GF(2^128)` that's
    /// ~7 multiplications + 127 squarings, versus the naive 126 mults
    /// + 127 squarings (an ~18× drop in multiplications). Worth doing
    /// once the PIOP starts hitting inverses on a hot path (currently
    /// the F_2 prove path doesn't; it's mostly mul + add).
    ///
    /// Panics if `self.is_zero()` — `0` has no multiplicative inverse.
    pub fn inverse(&self) -> Self {
        assert!(!self.is_zero(), "GF(2^128): zero has no inverse");
        let mut c = *self; // c = a^{2^1 - 1} = a
        for _ in 1..127 {
            c = c.square();
            c *= self;
        }
        // c = a^{2^127 - 1}. One more squaring → a^{2 · (2^127 - 1)} = a^{2^128 - 2}.
        c.square()
    }

    /// `self^exp` via binary square-and-multiply. `exp` is the natural-
    /// number exponent in `[0, 2^32)`.
    pub fn pow_u32(&self, mut exp: u32) -> Self {
        if exp == 0 {
            return Self::one();
        }
        let mut acc = Self::one();
        let mut base = *self;
        while exp > 0 {
            if exp & 1 == 1 {
                acc *= &base;
            }
            exp >>= 1;
            if exp > 0 {
                base = base.square();
            }
        }
        acc
    }
}

impl Zero for BinaryFieldGF128 {
    fn zero() -> Self {
        Self::zero()
    }
    fn is_zero(&self) -> bool {
        Self::is_zero(self)
    }
}

impl One for BinaryFieldGF128 {
    fn one() -> Self {
        Self::one()
    }
    fn is_one(&self) -> bool {
        let w = self.uint.as_words();
        w[0] == 1 && w[1] == 0
    }
}

impl Display for BinaryFieldGF128 {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let w = self.uint.as_words();
        write!(f, "GF128[{:016x}_{:016x}]", w[1], w[0])
    }
}

// -- additive group (XOR) --------------------------------------------

impl<'a> AddAssign<&'a Self> for BinaryFieldGF128 {
    #[inline]
    #[allow(clippy::arithmetic_side_effects, clippy::suspicious_op_assign_impl)]
    fn add_assign(&mut self, rhs: &'a Self) {
        let lw = *self.uint.as_words();
        let rw = rhs.uint.as_words();
        self.uint = Uint::<2>::from_words([lw[0] ^ rw[0], lw[1] ^ rw[1]]);
    }
}

impl AddAssign<Self> for BinaryFieldGF128 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        <Self as AddAssign<&Self>>::add_assign(self, &rhs);
    }
}

impl<'a> Add<&'a Self> for BinaryFieldGF128 {
    type Output = Self;
    #[inline]
    fn add(mut self, rhs: &'a Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl Add<Self> for BinaryFieldGF128 {
    type Output = Self;
    #[inline]
    fn add(mut self, rhs: Self) -> Self::Output {
        self += &rhs;
        self
    }
}

// In characteristic 2, subtraction == addition (XOR).
impl<'a> SubAssign<&'a Self> for BinaryFieldGF128 {
    #[inline]
    fn sub_assign(&mut self, rhs: &'a Self) {
        <Self as AddAssign<&Self>>::add_assign(self, rhs);
    }
}
impl SubAssign<Self> for BinaryFieldGF128 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self += rhs;
    }
}
impl<'a> Sub<&'a Self> for BinaryFieldGF128 {
    type Output = Self;
    #[inline]
    fn sub(mut self, rhs: &'a Self) -> Self::Output {
        self -= rhs;
        self
    }
}
impl Sub<Self> for BinaryFieldGF128 {
    type Output = Self;
    #[inline]
    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= &rhs;
        self
    }
}

impl Neg for BinaryFieldGF128 {
    type Output = Self;
    /// In characteristic 2, every element is its own additive inverse.
    #[inline]
    fn neg(self) -> Self::Output {
        self
    }
}

// -- multiplicative group --------------------------------------------

impl<'a> MulAssign<&'a Self> for BinaryFieldGF128 {
    #[inline]
    fn mul_assign(&mut self, rhs: &'a Self) {
        let product = clmul_128x128(self.uint.as_words(), rhs.uint.as_words());
        self.uint = Uint::<2>::from_words(reduce_256_to_128(product));
    }
}
impl MulAssign<Self> for BinaryFieldGF128 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        <Self as MulAssign<&Self>>::mul_assign(self, &rhs);
    }
}
impl<'a> Mul<&'a Self> for BinaryFieldGF128 {
    type Output = Self;
    #[inline]
    fn mul(mut self, rhs: &'a Self) -> Self::Output {
        self *= rhs;
        self
    }
}
impl Mul<Self> for BinaryFieldGF128 {
    type Output = Self;
    #[inline]
    fn mul(mut self, rhs: Self) -> Self::Output {
        self *= &rhs;
        self
    }
}

// -- division (multiply by inverse) ----------------------------------

impl<'a> DivAssign<&'a Self> for BinaryFieldGF128 {
    /// `self /= rhs` ≡ `self *= rhs.inverse()`. Panics if `rhs` is zero.
    #[inline]
    fn div_assign(&mut self, rhs: &'a Self) {
        *self *= rhs.inverse();
    }
}
impl DivAssign<Self> for BinaryFieldGF128 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self /= &rhs;
    }
}
impl<'a> Div<&'a Self> for BinaryFieldGF128 {
    type Output = Self;
    #[inline]
    fn div(mut self, rhs: &'a Self) -> Self::Output {
        self /= rhs;
        self
    }
}
impl Div<Self> for BinaryFieldGF128 {
    type Output = Self;
    #[inline]
    fn div(mut self, rhs: Self) -> Self::Output {
        self /= &rhs;
        self
    }
}

// -- num_traits checked-* (degenerate in a field — never fail) -------

impl CheckedAdd for BinaryFieldGF128 {
    #[inline]
    fn checked_add(&self, rhs: &Self) -> Option<Self> {
        Some(*self + rhs)
    }
}
impl CheckedSub for BinaryFieldGF128 {
    #[inline]
    fn checked_sub(&self, rhs: &Self) -> Option<Self> {
        Some(*self - rhs)
    }
}
impl CheckedMul for BinaryFieldGF128 {
    #[inline]
    fn checked_mul(&self, rhs: &Self) -> Option<Self> {
        Some(*self * rhs)
    }
}
impl CheckedNeg for BinaryFieldGF128 {
    #[inline]
    fn checked_neg(&self) -> Option<Self> {
        Some(-*self)
    }
}

// -- Sum / Product folds ---------------------------------------------

impl Sum<Self> for BinaryFieldGF128 {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}
impl<'a> Sum<&'a Self> for BinaryFieldGF128 {
    #[inline]
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + *x)
    }
}
impl Product<Self> for BinaryFieldGF128 {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}
impl<'a> Product<&'a Self> for BinaryFieldGF128 {
    #[inline]
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * *x)
    }
}

// -- Inv ------------------------------------------------------------

impl Inv for BinaryFieldGF128 {
    type Output = Option<Self>;
    #[inline]
    fn inv(self) -> Self::Output {
        if self.is_zero() {
            None
        } else {
            Some(self.inverse())
        }
    }
}

// -- Pow<u32> -------------------------------------------------------

impl Pow<u32> for BinaryFieldGF128 {
    type Output = Self;
    #[inline]
    fn pow(self, exp: u32) -> Self::Output {
        self.pow_u32(exp)
    }
}

// -- ConstZero / ConstOne / From<bool> ------------------------------

impl ConstZero for BinaryFieldGF128 {
    const ZERO: Self = Self::zero();
}

impl ConstOne for BinaryFieldGF128 {
    const ONE: Self = Self::one();
}

impl From<bool> for BinaryFieldGF128 {
    #[inline]
    fn from(b: bool) -> Self {
        if b { Self::one() } else { Self::zero() }
    }
}

// -- From<{u8..u128, i8..i128}> --------------------------------------
//
// Semantics: interpret the primitive's bit pattern as the low
// coefficients of a GF(2^128) element — i.e., `from(n: u64) = n`
// stored as bit-pattern words `[n, 0]`. The transcript only uses this
// map for hash absorption / challenge derivation, so any
// deterministic injection works; it just needs to be the same on
// prover and verifier.

macro_rules! impl_from_unsigned {
    ($($t:ty),*) => {
        $(
            impl From<$t> for BinaryFieldGF128 {
                #[inline]
                fn from(v: $t) -> Self {
                    Self::from_words([v as u64, 0])
                }
            }
        )*
    };
}

impl_from_unsigned!(u8, u16, u32, u64);

impl From<u128> for BinaryFieldGF128 {
    #[inline]
    fn from(v: u128) -> Self {
        Self::from_words([v as u64, (v >> 64) as u64])
    }
}

macro_rules! impl_from_signed {
    ($($t:ty),*) => {
        $(
            impl From<$t> for BinaryFieldGF128 {
                #[inline]
                fn from(v: $t) -> Self {
                    // Two's-complement bit pattern, sign-extended to u128.
                    Self::from(v as i128 as u128)
                }
            }
        )*
    };
}

impl_from_signed!(i8, i16, i32, i64);

impl From<i128> for BinaryFieldGF128 {
    #[inline]
    fn from(v: i128) -> Self {
        Self::from(v as u128)
    }
}

// -- Semiring / Ring marker traits ----------------------------------

impl Semiring for BinaryFieldGF128 {}
impl Ring for BinaryFieldGF128 {}

// -- Field ----------------------------------------------------------

impl Field for BinaryFieldGF128 {
    /// Bit-packed `Uint<2>` (128 bits, LSB-first within each `u64` limb).
    /// We use `Uint<2>` rather than `[u64; 2]` so that
    /// `F::Inner: ConstTranscribable` is satisfied directly (the
    /// transcript / Fiat-Shamir API uses `Inner` as the byte-level
    /// challenge representation).
    type Inner = Uint<2>;
    /// The reduction polynomial is hardcoded
    /// (`X^128 + X^7 + X^2 + X + 1`). Stored as the **low 128 bits**
    /// of `f(X)` — the `X^128` term is implicit, so the representable
    /// pattern is just `g(X) = X^7 + X^2 + X + 1` = `0x87`.
    type Modulus = Uint<2>;

    #[inline(always)]
    fn inner(&self) -> &Self::Inner {
        &self.uint
    }

    #[inline(always)]
    fn inner_mut(&mut self) -> &mut Self::Inner {
        &mut self.uint
    }

    #[inline(always)]
    fn into_inner(self) -> Self::Inner {
        self.uint
    }
}

// -- PrimeField (degenerate impl) -----------------------------------
//
// `BinaryFieldGF128` is a binary extension field, NOT a prime field.
// The codebase's `PrimeField` abstraction predates the F_2 work and is
// the central bound throughout `piop/` and `protocol/`. To reuse that
// machinery without a wide refactor, we implement `PrimeField`
// degenerately:
//
// - `Config = ()`: the field has no runtime-configurable data (the
//   reduction polynomial is compile-time-fixed).
// - `modulus()` / `make_cfg()`: return `MODULUS_LOW_BITS_GF128` / `Ok(())`.
// - `modulus_minus_one_div_two()`: **panics**. This method has no
//   meaningful analogue in characteristic 2; it is used by prime-field
//   square-root / Legendre-symbol code paths that are nonsense for
//   `GF(2^128)`. A panic is preferable to a silent wrong answer.

impl PrimeField for BinaryFieldGF128 {
    type Config = ();

    #[inline(always)]
    fn cfg(&self) -> &Self::Config {
        &()
    }

    #[inline(always)]
    fn is_zero(value: &Self) -> bool {
        Self::is_zero(value)
    }

    #[inline(always)]
    fn modulus(&self) -> Self::Modulus {
        MODULUS_LOW_BITS_GF128
    }

    fn modulus_minus_one_div_two(&self) -> Self::Inner {
        panic!(
            "BinaryFieldGF128::modulus_minus_one_div_two: GF(2^128) has no \
             prime modulus; this method is degenerate. Audit the call site \
             before using it with a binary extension field."
        );
    }

    fn make_cfg(modulus: &Self::Modulus) -> Result<Self::Config, FieldError> {
        if modulus == &MODULUS_LOW_BITS_GF128 {
            Ok(())
        } else {
            Err(FieldError::InvalidModulus)
        }
    }

    #[inline(always)]
    fn new_with_cfg(inner: Self::Inner, _cfg: &Self::Config) -> Self {
        // Input is a 128-bit `Uint<2>`; no high bits to reduce.
        Self { uint: inner }
    }

    #[inline(always)]
    fn new_unchecked_with_cfg(inner: Self::Inner, _cfg: &Self::Config) -> Self {
        Self { uint: inner }
    }

    #[inline(always)]
    fn zero_with_cfg(_cfg: &Self::Config) -> Self {
        Self::zero()
    }

    #[inline(always)]
    fn one_with_cfg(_cfg: &Self::Config) -> Self {
        Self::one()
    }
}

// -- InnerTransparentField -------------------------------------------
//
// `InnerTransparentField` requires field-arithmetic methods that
// operate directly on the inner-representation type. For
// `BinaryFieldGF128`, the inner repr IS the field-element repr
// (`Uint<2>` = bit-packed F_2[X]/<f>) — there's no Montgomery
// reinterpretation needed.

impl InnerTransparentField for BinaryFieldGF128 {
    #[inline]
    fn add_inner(
        lhs: &Self::Inner,
        rhs: &Self::Inner,
        _config: &Self::Config,
    ) -> Self::Inner {
        let lw = lhs.as_words();
        let rw = rhs.as_words();
        Uint::<2>::from_words([lw[0] ^ rw[0], lw[1] ^ rw[1]])
    }

    #[inline]
    fn sub_inner(
        lhs: &Self::Inner,
        rhs: &Self::Inner,
        config: &Self::Config,
    ) -> Self::Inner {
        // Characteristic 2: subtraction is XOR, same as addition.
        Self::add_inner(lhs, rhs, config)
    }

    #[inline]
    fn mul_assign_by_inner(&mut self, rhs: &Self::Inner) {
        // The inner representation is just the field element; lift to
        // `BinaryFieldGF128` and reuse the regular `MulAssign<&Self>`.
        let r = Self { uint: *rhs };
        *self *= &r;
    }
}

// -- carryless multiplication and reduction --------------------------
//
// 128×128 → 256-bit carryless product, computed as four 64×64 clmuls
// composed via Karatsuba (3 inner muls instead of 4). The 64×64 base is
// hardware-accelerated where available:
//   - aarch64: `vmull_p64` (PMULL/PMULL2)
//   - x86_64 + pclmulqdq: `_mm_clmulepi64_si128`
//   - else: a scalar fallback (bit-by-bit shift+XOR).

/// 128×128 → 256-bit carryless product.
///
/// Karatsuba layout (let `a = (a0, a1)`, `b = (b0, b1)`):
///   `lo`  = a0 · b0
///   `hi`  = a1 · b1
///   `mid` = (a0 ^ a1) · (b0 ^ b1) ^ lo ^ hi   = a0·b1 ^ a1·b0
/// Then the 256-bit product is `lo | (mid << 64) | (hi << 128)`.
#[inline]
fn clmul_128x128(a: &[u64; 2], b: &[u64; 2]) -> [u64; 4] {
    let lo = clmul_64x64(a[0], b[0]);
    let hi = clmul_64x64(a[1], b[1]);
    let mid_raw = clmul_64x64(a[0] ^ a[1], b[0] ^ b[1]);
    let mid = [mid_raw[0] ^ lo[0] ^ hi[0], mid_raw[1] ^ lo[1] ^ hi[1]];

    // Compose: word 0 = lo[0], word 1 = lo[1] ^ mid[0], word 2 = hi[0] ^ mid[1], word 3 = hi[1].
    [lo[0], lo[1] ^ mid[0], hi[0] ^ mid[1], hi[1]]
}

/// 64×64 → 128-bit carryless multiplication.
///
/// Dispatches to the hardware primitive when available, falls back to a
/// scalar bit-by-bit loop otherwise (used on platforms without PMULL /
/// PCLMUL; correctness mirror of the hardware path so tests stay
/// deterministic).
#[inline]
pub(crate) fn clmul_64x64(a: u64, b: u64) -> [u64; 2] {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: NEON+CRYPTO are enabled at compile time. `vmull_p64`
        // returns a 128-bit polynomial product as a `poly128_t`, which
        // we transmute to `[u64; 2]` (little-endian word order).
        unsafe {
            use core::arch::aarch64::vmull_p64;
            let prod: u128 = core::mem::transmute(vmull_p64(a, b));
            [prod as u64, (prod >> 64) as u64]
        }
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    {
        // SAFETY: pclmulqdq enabled at compile time.
        unsafe {
            use core::arch::x86_64::{__m128i, _mm_clmulepi64_si128, _mm_set_epi64x};
            let av = _mm_set_epi64x(0, a as i64);
            let bv = _mm_set_epi64x(0, b as i64);
            let prod: __m128i = _mm_clmulepi64_si128(av, bv, 0x00);
            let bytes: [u64; 2] = core::mem::transmute(prod);
            bytes
        }
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    )))]
    {
        clmul_64x64_scalar(a, b)
    }
}

/// Portable scalar fallback for 64×64 carryless mul. Correctness-only;
/// not exercised on the CI / bench targets (which all have PMULL or
/// PCLMUL).
#[inline]
#[allow(dead_code, clippy::arithmetic_side_effects)]
fn clmul_64x64_scalar(a: u64, b: u64) -> [u64; 2] {
    let mut a_word = a;
    let mut acc_lo: u64 = 0;
    let mut acc_hi: u64 = 0;
    while a_word != 0 {
        let lo = a_word.trailing_zeros();
        if lo == 0 {
            acc_lo ^= b;
        } else if lo < 64 {
            acc_lo ^= b << lo;
            acc_hi ^= b >> (64 - lo);
        }
        a_word &= a_word - 1;
    }
    [acc_lo, acc_hi]
}

/// Reduce a 256-bit `F_2[X]` polynomial modulo
/// `f(X) = X^128 + X^7 + X^2 + X + 1`.
///
/// For each bit `i ≥ 128` of the input, `X^i ≡ X^{i - 128} · g(X)`
/// (mod `f`), with `g(X) = X^7 + X^2 + X + 1`. Word-at-a-time:
///   - word 3 (bits 192..255) folds into `r[1]` (high half) and into
///     `hi2` for re-reduction;
///   - word 2 (bits 128..191) folds into `r[0]` and `r[1]`;
///   - `hi2`'s overflow (≤ 7 high bits → bits 128..134 globally)
///     re-folds once more into `r[0]`.
#[inline]
#[allow(clippy::arithmetic_side_effects)]
fn reduce_256_to_128(prod: [u64; 4]) -> [u64; 2] {
    let mut r = [prod[0], prod[1]];
    let mut hi2 = prod[2];

    // Step A: word 3 (bits 192..255). X^{192+k} ≡ X^{64+k} · g.
    let w = prod[3];
    let (lo, hi) = word_times_g_gf128(w);
    r[1] ^= lo;
    hi2 ^= hi;

    // Step B: word 2 (bits 128..191) — updated by step A's `hi`.
    // X^{128+k} ≡ X^k · g, so this lands in `r[0]` (bits 0..63) and
    // `r[1]` (bits 64..70 from the `<< 7` overflow).
    let w = hi2;
    let (lo, hi) = word_times_g_gf128(w);
    r[0] ^= lo;
    r[1] ^= hi;

    r
}

/// `w · g(X)` where `g(X) = X^7 + X^2 + X + 1`. Returns `(lo, hi)`.
#[inline]
#[allow(clippy::arithmetic_side_effects)]
fn word_times_g_gf128(w: u64) -> (u64, u64) {
    let lo = w ^ (w << 1) ^ (w << 2) ^ (w << 7);
    let hi = (w >> 63) ^ (w >> 62) ^ (w >> 57);
    (lo, hi)
}

// -- F_2[X]<D> → GF(2^128)[X]<D> coefficient lift --------------------
//
// Every `F_2` element is canonically a `GF(2^128)` element via the
// inclusion `F_2 ⊂ GF(2^128)`: `Boolean::ZERO → GF128::zero()` and
// `Boolean::ONE → GF128::one()`. Applied per-coefficient, this
// extends an `F_2[X]<D>`-typed cell to a `GF(2^128)[X]<D>` value
// the IC can then combine with `GF(2^128)`-valued challenges.

/// Lift `F_2[X]<D>` (a [`BinaryPoly<D>`]) to `GF(2^128)[X]<D>`.
/// Each `Boolean` coefficient maps to `GF128::zero()` or
/// `GF128::one()`.
pub fn lift_f2_poly_to_gf128<const D: usize>(p: &BinaryPoly<D>) -> GF128Poly<D> {
    let mut coeffs: [BinaryFieldGF128; D] = [BinaryFieldGF128::zero(); D];
    for (i, c) in p.iter().enumerate() {
        if c.inner() {
            coeffs[i] = BinaryFieldGF128::one();
        }
    }
    DensePolynomial { coeffs }
}

/// Lift `F_2[X]<D>` (a [`BinaryU64Poly<D>`], `D ≤ 64`) to
/// `GF(2^128)[X]<D>`. Same per-coefficient embedding as
/// [`lift_f2_poly_to_gf128`], but reading from the bit-packed
/// representation directly.
pub fn lift_f2_u64_poly_to_gf128<const D: usize>(p: &BinaryU64Poly<D>) -> GF128Poly<D> {
    assert!(D <= 64, "lift_f2_u64_poly_to_gf128: D ({D}) must be ≤ 64");
    let bits = *p.inner();
    let mut coeffs: [BinaryFieldGF128; D] = [BinaryFieldGF128::zero(); D];
    for (i, slot) in coeffs.iter_mut().enumerate().take(D) {
        #[allow(clippy::arithmetic_side_effects)]
        let bit = (bits >> i) & 1;
        if bit == 1 {
            *slot = BinaryFieldGF128::one();
        }
    }
    DensePolynomial { coeffs }
}

// -- evaluation: F_2[X]<D> → GF(2^128) at X = α ----------------------

/// Substitute `X = alpha` in an `F_2[X]<32>`-typed cell. See
/// [`eval_f2_poly_d_at`] for the `D`-generic version.
pub fn eval_f2_poly_d32_at(p: &BinaryPoly<32>, alpha: &BinaryFieldGF128) -> BinaryFieldGF128 {
    eval_f2_poly_d_at::<32>(p, alpha)
}

/// Substitute `X = alpha` in an `F_2[X]<D>`-typed cell. Each set bit
/// `i` of `p` adds `alpha^i` to the running accumulator. Requires
/// `D ≤ 64`.
pub fn eval_f2_poly_d_at<const D: usize>(
    p: &BinaryPoly<D>,
    alpha: &BinaryFieldGF128,
) -> BinaryFieldGF128 {
    assert!(D <= 64, "eval_f2_poly_d_at: D ({D}) must be ≤ 64");
    let mut bits: u64 = 0;
    for (i, c) in p.iter().enumerate() {
        if c.inner() {
            #[allow(clippy::arithmetic_side_effects)]
            {
                bits |= 1u64 << i;
            }
        }
    }
    eval_bits_at(bits, D, alpha)
}

/// Precompute the powers `α^0, α^1, ..., α^{d-1}` in `GF(2^128)`.
///
/// `d − 1` field multiplications. Each subsequent per-cell evaluation
/// becomes XOR-only (no inner-loop multiplications).
pub fn alpha_powers(alpha: &BinaryFieldGF128, d: usize) -> Vec<BinaryFieldGF128> {
    let mut powers = Vec::with_capacity(d);
    if d == 0 {
        return powers;
    }
    powers.push(BinaryFieldGF128::one());
    for _ in 1..d {
        let mut next = powers[powers.len() - 1];
        next *= &*alpha;
        powers.push(next);
    }
    powers
}

/// Evaluate an `F_2[X]<D>` cell at α using precomputed powers
/// (`alpha_powers[i] = α^i`). Inner loop is XOR-only: for each set bit
/// `i` of the cell, the accumulator XORs in `alpha_powers[i]`.
///
/// **Implementation note.** The hot loop uses the [`F2PackU64`] view of
/// the cell and walks the *set bits only* via `trailing_zeros` +
/// `bits &= bits - 1`. That skips zero bits and avoids the per-bit
/// branch the textbook form pays even when the bit is zero. The
/// accumulator is a raw `[u64; 2]` so the per-add work is two XORs
/// and a double load — no `BinaryFieldGF128`/`Uint<2>` rebuild per
/// iteration.
///
/// Caller must ensure `alpha_powers.len() >= D`.
#[inline]
#[allow(clippy::arithmetic_side_effects)]
pub fn eval_f2_poly_d_at_with_powers<const D: usize>(
    p: &BinaryPoly<D>,
    alpha_powers: &[BinaryFieldGF128],
) -> BinaryFieldGF128 {
    debug_assert!(
        alpha_powers.len() >= D,
        "eval_f2_poly_d_at_with_powers: powers slice ({}) shorter than D ({D})",
        alpha_powers.len(),
    );
    let mut bits = p.pack_u64();
    let mut acc = [0u64; 2];
    while bits != 0 {
        let i = bits.trailing_zeros() as usize;
        let pw = alpha_powers[i].words();
        acc[0] ^= pw[0];
        acc[1] ^= pw[1];
        bits &= bits - 1;
    }
    BinaryFieldGF128::from_words(acc)
}

/// Branchless variant of [`eval_f2_poly_d_at_with_powers`]: instead of
/// the `if bit_set` branch, mask each `alpha_powers[i]` by
/// `-(bit) as u64` and unconditionally XOR into the accumulator.
///
/// Caller must ensure `alpha_powers.len() >= D`.
#[inline]
pub fn eval_f2_poly_d_at_with_powers_branchless<const D: usize>(
    p: &BinaryPoly<D>,
    alpha_powers: &[BinaryFieldGF128],
) -> BinaryFieldGF128 {
    debug_assert!(
        alpha_powers.len() >= D,
        "eval_f2_poly_d_at_with_powers_branchless: powers slice ({}) shorter than D ({D})",
        alpha_powers.len(),
    );
    let mut acc_lo: u64 = 0;
    let mut acc_hi: u64 = 0;
    for (i, c) in p.iter().enumerate() {
        let mask: u64 = 0u64.wrapping_sub(c.inner() as u64);
        let pw = alpha_powers[i].words();
        acc_lo ^= pw[0] & mask;
        acc_hi ^= pw[1] & mask;
    }
    BinaryFieldGF128::from_words([acc_lo, acc_hi])
}

/// SIMD-batched α-projection: evaluate four `F_2[X]<D>`-typed cells at
/// the shared α (`alpha_powers[i] = α^i`) in a single pass.
///
/// Motivation. The branchy single-cell kernel
/// [`eval_f2_poly_d_at_with_powers`] is loop-control-bound on Apple
/// Silicon: each set bit pays a `trailing_zeros` + `bits &= bits - 1`
/// + α-load + 2 XOR sequence (~5–6 ops), and the load + XOR is too
/// small to hide the loop overhead. Batching four cells in the
/// branchless form lets a single α-load amortise across four
/// accumulators, and (on aarch64) lets us use one NEON `veorq_u64`
/// per 16-byte accumulator update instead of two scalar XORs.
///
/// Layout. The body is the `D`-iteration **branchless** form (no
/// data-dependent branches; cost is fixed `D × work` regardless of
/// bit pattern). The mask per cell is `0u64.wrapping_sub((bits >> i) & 1)`
/// — `0` if the cell has bit `i` clear, `!0` if set — so the
/// `pw & mask` either contributes `pw` to that cell's accumulator
/// or contributes `0`. This is correctness-equivalent to the branchy
/// form (each cell ends up XOR-ing in the same set of `α^i`).
///
/// `cells` carries the four bit-packed cells (`BinaryPoly::pack_u64()`
/// of each). Callers should batch their column into chunks of four
/// and call this once per chunk; see the prove-path call site in
/// `protocol/src/f2_prove.rs`.
///
/// Caller must ensure `alpha_powers.len() >= D` and `D ≤ 64`.
#[inline]
#[allow(clippy::arithmetic_side_effects)]
pub fn eval_f2_poly_d_at_with_powers_simd_x4<const D: usize>(
    cells: [u64; 4],
    alpha_powers: &[BinaryFieldGF128],
) -> [BinaryFieldGF128; 4] {
    debug_assert!(
        alpha_powers.len() >= D,
        "eval_f2_poly_d_at_with_powers_simd_x4: powers slice ({}) shorter than D ({D})",
        alpha_powers.len(),
    );
    debug_assert!(
        D <= 64,
        "eval_f2_poly_d_at_with_powers_simd_x4: D ({D}) must be ≤ 64"
    );

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        use core::arch::aarch64::{vandq_u64, vdupq_n_u64, veorq_u64, vld1q_u64, vst1q_u64};
        // SAFETY: NEON enabled at compile time; we hold the four
        // accumulators in independent registers and only ever read 16
        // bytes from a `BinaryFieldGF128` (which is `#[repr(transparent)]`
        // over `Uint<2>`, i.e. `[u64; 2]`).
        unsafe {
            let mut a0 = vdupq_n_u64(0);
            let mut a1 = vdupq_n_u64(0);
            let mut a2 = vdupq_n_u64(0);
            let mut a3 = vdupq_n_u64(0);

            for i in 0..D {
                let pw = vld1q_u64(alpha_powers[i].words().as_ptr());

                let m0 = vdupq_n_u64(0u64.wrapping_sub((cells[0] >> i) & 1));
                let m1 = vdupq_n_u64(0u64.wrapping_sub((cells[1] >> i) & 1));
                let m2 = vdupq_n_u64(0u64.wrapping_sub((cells[2] >> i) & 1));
                let m3 = vdupq_n_u64(0u64.wrapping_sub((cells[3] >> i) & 1));

                a0 = veorq_u64(a0, vandq_u64(pw, m0));
                a1 = veorq_u64(a1, vandq_u64(pw, m1));
                a2 = veorq_u64(a2, vandq_u64(pw, m2));
                a3 = veorq_u64(a3, vandq_u64(pw, m3));
            }

            let mut w0 = [0u64; 2];
            let mut w1 = [0u64; 2];
            let mut w2 = [0u64; 2];
            let mut w3 = [0u64; 2];
            vst1q_u64(w0.as_mut_ptr(), a0);
            vst1q_u64(w1.as_mut_ptr(), a1);
            vst1q_u64(w2.as_mut_ptr(), a2);
            vst1q_u64(w3.as_mut_ptr(), a3);

            [
                BinaryFieldGF128::from_words(w0),
                BinaryFieldGF128::from_words(w1),
                BinaryFieldGF128::from_words(w2),
                BinaryFieldGF128::from_words(w3),
            ]
        }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        // Portable scalar fallback — same algorithm, 4 cells in
        // parallel sharing the α^i load, but with regular u64 XOR/AND
        // instead of NEON intrinsics. Loses the NEON dispatch advantage
        // but still amortises the α load across 4 cells.
        let mut a0 = [0u64; 2];
        let mut a1 = [0u64; 2];
        let mut a2 = [0u64; 2];
        let mut a3 = [0u64; 2];
        for i in 0..D {
            let pw = alpha_powers[i].words();
            let m0 = 0u64.wrapping_sub((cells[0] >> i) & 1);
            let m1 = 0u64.wrapping_sub((cells[1] >> i) & 1);
            let m2 = 0u64.wrapping_sub((cells[2] >> i) & 1);
            let m3 = 0u64.wrapping_sub((cells[3] >> i) & 1);
            a0[0] ^= pw[0] & m0;
            a0[1] ^= pw[1] & m0;
            a1[0] ^= pw[0] & m1;
            a1[1] ^= pw[1] & m1;
            a2[0] ^= pw[0] & m2;
            a2[1] ^= pw[1] & m2;
            a3[0] ^= pw[0] & m3;
            a3[1] ^= pw[1] & m3;
        }
        [
            BinaryFieldGF128::from_words(a0),
            BinaryFieldGF128::from_words(a1),
            BinaryFieldGF128::from_words(a2),
            BinaryFieldGF128::from_words(a3),
        ]
    }
}

/// Project a slice of `F_2[X]<D>` cells at the shared α using
/// [`eval_f2_poly_d_at_with_powers_simd_x4`] for groups of 4, falling
/// back to the scalar branchy kernel for the remainder.
///
/// Convenience wrapper for the prove-path column projection: takes a
/// slice of cells (`column.evaluations`), returns the per-cell
/// `GF(2^128)` evaluation. Caller pre-computes `alpha_powers`.
#[inline]
pub fn project_column_with_powers<const D: usize>(
    cells: &[BinaryPoly<D>],
    alpha_powers: &[BinaryFieldGF128],
) -> Vec<BinaryFieldGF128> {
    debug_assert!(
        alpha_powers.len() >= D,
        "project_column_with_powers: powers slice ({}) shorter than D ({D})",
        alpha_powers.len(),
    );
    let mut out = Vec::with_capacity(cells.len());
    let mut chunks = cells.chunks_exact(4);
    for chunk in &mut chunks {
        let packed = [
            chunk[0].pack_u64(),
            chunk[1].pack_u64(),
            chunk[2].pack_u64(),
            chunk[3].pack_u64(),
        ];
        let r = eval_f2_poly_d_at_with_powers_simd_x4::<D>(packed, alpha_powers);
        out.extend_from_slice(&r);
    }
    for cell in chunks.remainder() {
        out.push(eval_f2_poly_d_at_with_powers::<D>(cell, alpha_powers));
    }
    out
}

/// IC-style accumulator: for each row `i` in `0..cells.len()` and each
/// set bit `d` of `cells[i]`, XOR `eq_table[i]` into `coeffs[d]`.
///
/// This is the inner loop of `F2NativeIc::prove_linear`'s up_evals /
/// down_evals. Specialised for `BinaryFieldGF128` so the per-bit
/// 128-bit XOR uses NEON `veorq_u64` directly (rather than relying on
/// the compiler to vectorise two `u64` XORs through the
/// `Uint::<2>::from_words` round-trip in the generic `AddAssign` impl).
///
/// Caller guarantees:
///   * `cells.len() == eq_table.len()` (the kernel walks `cells.len()` rows).
///   * `coeffs.len() >= D`.
///   * `D <= 64`.
#[inline]
#[allow(clippy::arithmetic_side_effects)]
pub fn ic_accumulate_into_coeffs_bf128<const D: usize>(
    cells: &[BinaryPoly<D>],
    eq_table: &[BinaryFieldGF128],
    coeffs: &mut [BinaryFieldGF128],
) {
    debug_assert!(D <= 64);
    debug_assert_eq!(cells.len(), eq_table.len());
    debug_assert!(coeffs.len() >= D);

    let n = cells.len();

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        use core::arch::aarch64::{veorq_u64, vld1q_u64, vst1q_u64};
        // SAFETY: NEON enabled at compile time; we only touch
        // `BinaryFieldGF128` slots through their `#[repr(transparent)]
        // Uint<2>` layout (== `[u64; 2]`).
        unsafe {
            for i in 0..n {
                let mut bits = cells.get_unchecked(i).pack_u64();
                if bits == 0 {
                    continue;
                }
                let eq_q = vld1q_u64(eq_table.get_unchecked(i).words().as_ptr());
                while bits != 0 {
                    let d = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    // SAFETY: `d < D` since `bits` has bits only in
                    // 0..D after `pack_u64`'s D-mask; `coeffs.len() >= D`
                    // by caller invariant.
                    let slot = coeffs.get_unchecked_mut(d);
                    let p = vld1q_u64(slot.words().as_ptr());
                    let v = veorq_u64(p, eq_q);
                    // We need a *mut [u64;2] writeable view of the
                    // slot. The slot is repr(transparent) over Uint<2>
                    // = [u64; 2], so a cast to *mut u64 is well-defined.
                    vst1q_u64(slot as *mut BinaryFieldGF128 as *mut u64, v);
                }
            }
        }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        for i in 0..n {
            let mut bits = cells[i].pack_u64();
            if bits == 0 {
                continue;
            }
            let eq_words = *eq_table[i].words();
            while bits != 0 {
                let d = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let cur = *coeffs[d].words();
                coeffs[d] = BinaryFieldGF128::from_words([
                    cur[0] ^ eq_words[0],
                    cur[1] ^ eq_words[1],
                ]);
            }
        }
    }
}

/// 4-column-batched variant of [`ic_accumulate_into_coeffs_bf128`].
/// Processes four columns in lock-step over the same row range so
/// `eq_table[i]` is loaded once per row and reused across all four
/// per-column bit-walks. This is the structural win: in the original
/// per-column loop the `eq_table` (typically 64 MB at SHA-256 nvars=22)
/// is streamed once per column (41 streams), so 41 × 64 MB ≈ 2.6 GB of
/// eq-table bandwidth. With 4-col batching it's streamed once per
/// 4-col chunk, dropping to ~700 MB.
///
/// All four `cells_*` slices must have the same length as `eq_table`.
/// All four `coeffs_*` slices must have length `>= D`.
#[inline]
#[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
pub fn ic_accumulate_4cols_simd_x4_bf128<const D: usize>(
    cells_a: &[BinaryPoly<D>],
    cells_b: &[BinaryPoly<D>],
    cells_c: &[BinaryPoly<D>],
    cells_d: &[BinaryPoly<D>],
    eq_table: &[BinaryFieldGF128],
    coeffs_a: &mut [BinaryFieldGF128],
    coeffs_b: &mut [BinaryFieldGF128],
    coeffs_c: &mut [BinaryFieldGF128],
    coeffs_d: &mut [BinaryFieldGF128],
) {
    debug_assert!(D <= 64);
    let n = eq_table.len();
    debug_assert_eq!(cells_a.len(), n);
    debug_assert_eq!(cells_b.len(), n);
    debug_assert_eq!(cells_c.len(), n);
    debug_assert_eq!(cells_d.len(), n);
    debug_assert!(coeffs_a.len() >= D);
    debug_assert!(coeffs_b.len() >= D);
    debug_assert!(coeffs_c.len() >= D);
    debug_assert!(coeffs_d.len() >= D);

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        use core::arch::aarch64::{veorq_u64, vld1q_u64, vst1q_u64};

        // Per-column branchy walk inside the shared-eq row loop.
        // Hoisting this as a local closure (#[inline(always)] via the
        // outer `#[inline]`) keeps the branch-prediction trace per-
        // column small while still inlining at the call sites.
        //
        // SAFETY for the inner unsafe block:
        //   * `d` is at most 63 (D <= 64 + pack_u64's D-mask), and
        //     `coeffs.len() >= D` by caller invariant.
        //   * `slot` is `*mut BinaryFieldGF128`; its underlying storage
        //     is `repr(transparent)` over `[u64; 2]`, so reading and
        //     writing 16 bytes via `vld1q_u64` / `vst1q_u64` is sound.
        let walk = |mut bits: u64, eq_q: core::arch::aarch64::uint64x2_t,
                    coeffs: &mut [BinaryFieldGF128]| unsafe {
            while bits != 0 {
                let d = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let slot = coeffs.get_unchecked_mut(d);
                let p = vld1q_u64(slot.words().as_ptr());
                let v = veorq_u64(p, eq_q);
                vst1q_u64(slot as *mut BinaryFieldGF128 as *mut u64, v);
            }
        };

        // SAFETY: as the inner closure, plus all four cell slices were
        // asserted to have length `n` == eq_table.len().
        unsafe {
            for i in 0..n {
                let a_bits = cells_a.get_unchecked(i).pack_u64();
                let b_bits = cells_b.get_unchecked(i).pack_u64();
                let c_bits = cells_c.get_unchecked(i).pack_u64();
                let d_bits = cells_d.get_unchecked(i).pack_u64();
                if (a_bits | b_bits | c_bits | d_bits) == 0 {
                    continue;
                }
                let eq_q = vld1q_u64(eq_table.get_unchecked(i).words().as_ptr());
                walk(a_bits, eq_q, coeffs_a);
                walk(b_bits, eq_q, coeffs_b);
                walk(c_bits, eq_q, coeffs_c);
                walk(d_bits, eq_q, coeffs_d);
            }
        }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        // Scalar fallback: 4 columns processed in lock-step, eq_words
        // loaded once per row. Auto-vectorisation may collapse the
        // two-u64 XOR into a single 128-bit op on x86_64+SSE2.
        for i in 0..n {
            let a_bits = cells_a[i].pack_u64();
            let b_bits = cells_b[i].pack_u64();
            let c_bits = cells_c[i].pack_u64();
            let d_bits = cells_d[i].pack_u64();
            if (a_bits | b_bits | c_bits | d_bits) == 0 {
                continue;
            }
            let eq_words = *eq_table[i].words();
            let walk = |mut bits: u64, coeffs: &mut [BinaryFieldGF128]| {
                while bits != 0 {
                    let dd = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let cur = *coeffs[dd].words();
                    coeffs[dd] = BinaryFieldGF128::from_words([
                        cur[0] ^ eq_words[0],
                        cur[1] ^ eq_words[1],
                    ]);
                }
            };
            walk(a_bits, coeffs_a);
            walk(b_bits, coeffs_b);
            walk(c_bits, coeffs_c);
            walk(d_bits, coeffs_d);
        }
    }
}

/// Sparse-skip variant of [`eval_f2_poly_d_at_with_powers_simd_x4`]:
/// walk only positions where **at least one** of the four cells has
/// the bit set (i.e. iterate over `cells[0] | cells[1] | cells[2] | cells[3]`
/// via `trailing_zeros` instead of `0..D`).
///
/// Trade-off vs the dense `simd_x4` variant: each iteration carries
/// the loop control (CTZ + `bits &= bits - 1` + branch) plus the same
/// NEON 4-way XOR body. Sparse wins when the union's popcount is
/// substantially less than `D` — the case on SHA-256-style traces
/// where many cells have low popcount and the union still stays well
/// below 32. Dense `simd_x4` wins on random / high-popcount inputs
/// where the union saturates near `D`.
///
/// Caller must ensure `alpha_powers.len() >= D` and `D ≤ 64`.
#[inline]
#[allow(clippy::arithmetic_side_effects)]
pub fn eval_f2_poly_d_at_with_powers_simd_x4_sparse<const D: usize>(
    cells: [u64; 4],
    alpha_powers: &[BinaryFieldGF128],
) -> [BinaryFieldGF128; 4] {
    debug_assert!(
        alpha_powers.len() >= D,
        "eval_f2_poly_d_at_with_powers_simd_x4_sparse: powers slice ({}) shorter than D ({D})",
        alpha_powers.len(),
    );
    debug_assert!(D <= 64);

    // Mask the union to bits 0..D so a caller-supplied junk-high-bit
    // pattern doesn't make us read past `alpha_powers[D-1]`.
    let mask_d: u64 = if D == 64 { !0u64 } else { (1u64 << D) - 1 };
    let union_bits = (cells[0] | cells[1] | cells[2] | cells[3]) & mask_d;

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        use core::arch::aarch64::{vandq_u64, vdupq_n_u64, veorq_u64, vld1q_u64, vst1q_u64};
        // SAFETY: as in `_simd_x4`, NEON enabled at compile time.
        unsafe {
            let mut a0 = vdupq_n_u64(0);
            let mut a1 = vdupq_n_u64(0);
            let mut a2 = vdupq_n_u64(0);
            let mut a3 = vdupq_n_u64(0);

            let mut bits = union_bits;
            while bits != 0 {
                let i = bits.trailing_zeros() as usize;
                let pw = vld1q_u64(alpha_powers[i].words().as_ptr());

                let m0 = vdupq_n_u64(0u64.wrapping_sub((cells[0] >> i) & 1));
                let m1 = vdupq_n_u64(0u64.wrapping_sub((cells[1] >> i) & 1));
                let m2 = vdupq_n_u64(0u64.wrapping_sub((cells[2] >> i) & 1));
                let m3 = vdupq_n_u64(0u64.wrapping_sub((cells[3] >> i) & 1));

                a0 = veorq_u64(a0, vandq_u64(pw, m0));
                a1 = veorq_u64(a1, vandq_u64(pw, m1));
                a2 = veorq_u64(a2, vandq_u64(pw, m2));
                a3 = veorq_u64(a3, vandq_u64(pw, m3));

                bits &= bits - 1;
            }

            let mut w0 = [0u64; 2];
            let mut w1 = [0u64; 2];
            let mut w2 = [0u64; 2];
            let mut w3 = [0u64; 2];
            vst1q_u64(w0.as_mut_ptr(), a0);
            vst1q_u64(w1.as_mut_ptr(), a1);
            vst1q_u64(w2.as_mut_ptr(), a2);
            vst1q_u64(w3.as_mut_ptr(), a3);

            [
                BinaryFieldGF128::from_words(w0),
                BinaryFieldGF128::from_words(w1),
                BinaryFieldGF128::from_words(w2),
                BinaryFieldGF128::from_words(w3),
            ]
        }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        let mut a0 = [0u64; 2];
        let mut a1 = [0u64; 2];
        let mut a2 = [0u64; 2];
        let mut a3 = [0u64; 2];
        let mut bits = union_bits;
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            let pw = alpha_powers[i].words();
            let m0 = 0u64.wrapping_sub((cells[0] >> i) & 1);
            let m1 = 0u64.wrapping_sub((cells[1] >> i) & 1);
            let m2 = 0u64.wrapping_sub((cells[2] >> i) & 1);
            let m3 = 0u64.wrapping_sub((cells[3] >> i) & 1);
            a0[0] ^= pw[0] & m0;
            a0[1] ^= pw[1] & m0;
            a1[0] ^= pw[0] & m1;
            a1[1] ^= pw[1] & m1;
            a2[0] ^= pw[0] & m2;
            a2[1] ^= pw[1] & m2;
            a3[0] ^= pw[0] & m3;
            a3[1] ^= pw[1] & m3;
            bits &= bits - 1;
        }
        [
            BinaryFieldGF128::from_words(a0),
            BinaryFieldGF128::from_words(a1),
            BinaryFieldGF128::from_words(a2),
            BinaryFieldGF128::from_words(a3),
        ]
    }
}

/// Sparse-variant column projector — mirror of
/// [`project_column_with_powers`] but routing batches through the
/// union-bit-skipping kernel
/// [`eval_f2_poly_d_at_with_powers_simd_x4_sparse`]. Remainder cells
/// still go through the scalar branchy fallback.
#[inline]
pub fn project_column_with_powers_sparse<const D: usize>(
    cells: &[BinaryPoly<D>],
    alpha_powers: &[BinaryFieldGF128],
) -> Vec<BinaryFieldGF128> {
    debug_assert!(
        alpha_powers.len() >= D,
        "project_column_with_powers_sparse: powers slice ({}) shorter than D ({D})",
        alpha_powers.len(),
    );
    let mut out = Vec::with_capacity(cells.len());
    let mut chunks = cells.chunks_exact(4);
    for chunk in &mut chunks {
        let packed = [
            chunk[0].pack_u64(),
            chunk[1].pack_u64(),
            chunk[2].pack_u64(),
            chunk[3].pack_u64(),
        ];
        let r = eval_f2_poly_d_at_with_powers_simd_x4_sparse::<D>(packed, alpha_powers);
        out.extend_from_slice(&r);
    }
    for cell in chunks.remainder() {
        out.push(eval_f2_poly_d_at_with_powers::<D>(cell, alpha_powers));
    }
    out
}

/// Substitute `X = alpha` in an `F_2[X]<D>`-typed cell stored in
/// [`BinaryU64Poly`] form. `D` must be ≤ 64.
pub fn eval_f2_u64_poly_at<const D: usize>(
    p: &BinaryU64Poly<D>,
    alpha: &BinaryFieldGF128,
) -> BinaryFieldGF128 {
    assert!(D <= 64, "eval_f2_u64_poly_at: D ({D}) must be ≤ 64");
    eval_bits_at(*p.inner(), D, alpha)
}

/// Substitute `X = alpha` in a [`BinaryF2Poly<W>`] (wide F_2[X]
/// stored as `[u64; W]`).
#[allow(clippy::arithmetic_side_effects)]
pub fn eval_f2_wide_poly_at<const W: usize>(
    p: &BinaryF2Poly<W>,
    alpha: &BinaryFieldGF128,
) -> BinaryFieldGF128 {
    let mut acc = BinaryFieldGF128::zero();
    let mut pow = BinaryFieldGF128::one(); // α^0
    let mut idx = 0usize;
    let words = p.words();
    for w_idx in 0..W {
        let mut w = words[w_idx];
        while w != 0 {
            let lsb = w.trailing_zeros() as usize;
            let target = w_idx * 64 + lsb;
            while idx < target {
                pow *= alpha;
                idx += 1;
            }
            acc += &pow;
            w &= w - 1;
        }
    }
    acc
}

// -- F_2[X] ↔ GF(2^128) representative lifts ------------------------

// -- α-dependent inverse lift ---------------------------------------
//
// The "lift" that makes the F_2[X] open work for a transcript-fresh
// α is **not** the bit-identical canonical representative — see the
// note in `f2_open_plan.md` § Risks. Bit-identical only satisfies
// `ψ_α(lift(g)) = g` when α is the field's quotient generator `X`;
// for generic α, we need the unique polynomial in `F_2[X]<128>`
// whose evaluation at α equals g.
//
// Concretely: writing g ∈ GF(2^128) as a 128-bit vector g_bits, and
// noting that {1, α, α^2, …, α^{127}} forms an F_2-basis of
// GF(2^128) precisely when α has minimal polynomial of degree 128
// over F_2 (probability ≥ 1 − 2^{-63} for transcript-fresh α), the
// lift solves the linear system
//
//   Σ_j c_j · α^j  =  g       (in GF(2^128) arithmetic),
//
// for the coefficient vector c = (c_0, …, c_{127}) ∈ F_2^{128}.
// `q' = Σ_j c_j X^j ∈ F_2[X]<128>` is then the unique element with
// `ψ_α(q') = g`.

/// Precomputed lift table for a fixed α. Building it costs ~128
/// GF(2^128) multiplications + one 128×128 F_2 matrix inversion;
/// individual lifts then cost 128 word XORs each.
///
/// Panics during construction if α has minimal polynomial of degree
/// strictly less than 128 over F_2 (the basis `{α^j}` is not
/// invertible). For transcript-fresh α this happens with negligible
/// probability.
pub struct AlphaPolyBasis {
    /// `inverse[i]` is row i of `M_α^{-1}` packed as a 128-bit
    /// vector (2 × `u64`, LSB-first per limb). Used to compute the
    /// lift coefficient `c_i = <inverse[i], g_bits>` over F_2.
    inverse: [[u64; 2]; 128],
}

impl AlphaPolyBasis {
    /// Build the lift table for a given α. Panics if α's minimal
    /// polynomial over F_2 has degree < 128.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn new(alpha: &BinaryFieldGF128) -> Self {
        // M_α has columns α^0, α^1, …, α^{127}. We store it
        // row-by-row so the inversion routine can pivot on rows.
        let mut m_rows = [[0u64; 2]; 128];
        let mut cur = BinaryFieldGF128::one(); // α^0 = 1
        for j in 0..128 {
            let cw = cur.words();
            // Bit i of α^j contributes to row i, column j.
            for i in 0..128 {
                let bit = (cw[i / 64] >> (i % 64)) & 1;
                if bit == 1 {
                    m_rows[i][j / 64] |= 1u64 << (j % 64);
                }
            }
            if j + 1 < 128 {
                cur *= alpha;
            }
        }
        let inverse = invert_f2_matrix_128(&m_rows);
        Self { inverse }
    }

    /// Lift `g ∈ GF(2^128)` to `BinaryF2Poly<2>`. The result `q'`
    /// satisfies `ψ_α(q') = g`, i.e. `Σ_i q'_i · α^i = g`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn lift(&self, g: &BinaryFieldGF128) -> BinaryF2Poly<2> {
        let g_bits = *g.words();
        let mut out = [0u64; 2];
        for i in 0..128 {
            // F_2 inner product of `inverse[i]` (a 128-bit row) and
            // `g_bits` (a 128-bit vector).
            let mut acc = 0u64;
            for w in 0..2 {
                acc ^= self.inverse[i][w] & g_bits[w];
            }
            let bit = (acc.count_ones() & 1) as u64;
            out[i / 64] |= bit << (i % 64);
        }
        BinaryF2Poly::<2>::from_words(out)
    }
}

/// Invert a 128×128 F_2 matrix stored row-by-row (each row as 2 ×
/// `u64`). Uses Gauss-Jordan with an augmented `[A | I]` matrix.
/// Panics if `A` is singular over F_2.
#[allow(clippy::arithmetic_side_effects)]
fn invert_f2_matrix_128(rows: &[[u64; 2]; 128]) -> [[u64; 2]; 128] {
    // Augmented: 4 u64 per row — left 2 = A, right 2 = I (initially).
    let mut m: [[u64; 4]; 128] = [[0u64; 4]; 128];
    for i in 0..128 {
        m[i][0] = rows[i][0];
        m[i][1] = rows[i][1];
        // Right half = identity: bit i of row i in the right block.
        m[i][2 + i / 64] = 1u64 << (i % 64);
    }

    for col in 0..128 {
        // Find a pivot row at or below `col` with bit `col` set in
        // the left block.
        let mut piv = None;
        for r in col..128 {
            if (m[r][col / 64] >> (col % 64)) & 1 == 1 {
                piv = Some(r);
                break;
            }
        }
        let piv = piv.expect(
            "alpha basis is singular: α has minimal polynomial of degree < 128 over F_2",
        );
        if piv != col {
            m.swap(piv, col);
        }
        // Eliminate `col` in every other row.
        for r in 0..128 {
            if r == col {
                continue;
            }
            if (m[r][col / 64] >> (col % 64)) & 1 == 1 {
                for w in 0..4 {
                    m[r][w] ^= m[col][w];
                }
            }
        }
    }

    let mut inv = [[0u64; 2]; 128];
    for i in 0..128 {
        inv[i][0] = m[i][2];
        inv[i][1] = m[i][3];
    }
    inv
}

/// Lift a `BinaryPoly<D>` (`D ≤ 64`) into `BinaryF2Poly<1>` by
/// packing its bits into a single `u64`.
///
/// Inverse of "read the bottom `D` bits and treat as a
/// `BinaryPoly<D>`": `lift_bp_to_f2_poly_1(p).words()[0] & ((1 << D)
/// - 1)` matches the bit pattern of `p`. Required for the F_2[X]
/// inner-product helpers, which operate on `BinaryF2Poly<W>` so the
/// product widths can grow with `W`.
#[allow(clippy::arithmetic_side_effects)]
#[inline]
pub fn lift_bp_to_f2_poly_1<const D: usize>(p: &BinaryPoly<D>) -> BinaryF2Poly<1> {
    assert!(D <= 64, "lift_bp_to_f2_poly_1: D ({D}) must be ≤ 64");
    let bits = p.pack_u64();
    let masked = if D == 64 { bits } else { bits & ((1u64 << D) - 1) };
    BinaryF2Poly::<1>::from_words([masked])
}

/// Bit-pattern (canonical-representative) embedding `GF(2^128) →
/// F_2[X]<128>` as `BinaryF2Poly<2>`.
///
/// **Caveat**: `ψ_α(lift_gf128_to_f2_poly_2(g)) = g` holds **only**
/// when α is the field's quotient generator `X` (mod `P`). For a
/// transcript-fresh α, use [`AlphaPolyBasis::lift`] instead.
///
/// This helper is kept for completeness and for callers that
/// genuinely want the canonical representative (e.g., interpreting
/// a field element's word storage as a `BinaryF2Poly` for transcript
/// serialisation).
#[inline]
pub fn lift_gf128_to_f2_poly_2(g: &BinaryFieldGF128) -> BinaryF2Poly<2> {
    BinaryF2Poly::<2>::from_words(*g.words())
}

/// Project a `BinaryF2Poly<2>` back into `GF(2^128)` by reading its
/// bits as the canonical degree-<128 representative. The inverse of
/// [`lift_gf128_to_f2_poly_2`] when the input has no bits set at or
/// above position 128 (the natural invariant for `BinaryF2Poly<2>`).
#[inline]
pub fn f2_poly_2_to_gf128(p: &BinaryF2Poly<2>) -> BinaryFieldGF128 {
    BinaryFieldGF128::from_words(*p.words())
}

/// Inner kernel for `eval_*` variants whose `<D>` polynomial fits in a
/// `u64`. Walks bits of `bits` from LSB up, multiplying a running
/// `pow = α^i` by `alpha` at each step.
#[inline]
#[allow(clippy::arithmetic_side_effects)]
fn eval_bits_at(mut bits: u64, max_bits: usize, alpha: &BinaryFieldGF128) -> BinaryFieldGF128 {
    debug_assert!(max_bits <= 64);
    let mut acc = BinaryFieldGF128::zero();
    let mut pow = BinaryFieldGF128::one(); // α^0
    let mut idx = 0usize;
    while bits != 0 && idx < max_bits {
        if bits & 1 == 1 {
            acc += &pow;
        }
        bits >>= 1;
        if bits != 0 && idx + 1 < max_bits {
            pow *= alpha;
        }
        idx += 1;
    }
    acc
}

// -- tests -----------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    fn rand_elt(rng: &mut StdRng) -> BinaryFieldGF128 {
        BinaryFieldGF128::from_words([rng.random(), rng.random()])
    }

    fn gf(lo: u64, hi: u64) -> BinaryFieldGF128 {
        BinaryFieldGF128::from_words([lo, hi])
    }

    #[test]
    fn zero_and_one_are_identities() {
        let z = BinaryFieldGF128::zero();
        let o = BinaryFieldGF128::one();
        let a = gf(0xDEAD_BEEF_CAFE_F00D, 0xA5A5_A5A5_A5A5_A5A5);
        assert_eq!(a + z, a);
        assert_eq!(a * o, a);
        assert_eq!(a * z, z);
    }

    #[test]
    fn add_is_xor_and_self_inverse() {
        let mut rng = StdRng::seed_from_u64(0xC0FFEE_128);
        for _ in 0..256 {
            let a = rand_elt(&mut rng);
            let b = rand_elt(&mut rng);
            assert_eq!((a + b) + b, a); // char 2: x + x = 0
            assert_eq!(a + BinaryFieldGF128::zero(), a);
        }
    }

    #[test]
    fn one_is_mul_identity_and_zero_annihilates() {
        let mut rng = StdRng::seed_from_u64(0xDEAD_128);
        let one = BinaryFieldGF128::one();
        let zero = BinaryFieldGF128::zero();
        for _ in 0..256 {
            let a = rand_elt(&mut rng);
            assert_eq!(a * one, a);
            assert_eq!(a * zero, zero);
        }
    }

    #[test]
    fn square_matches_self_multiply() {
        let mut rng = StdRng::seed_from_u64(0xBEEF_128);
        for _ in 0..256 {
            let a = rand_elt(&mut rng);
            assert_eq!(a.square(), a * a);
        }
    }

    #[test]
    fn multiplication_is_commutative_and_associative() {
        let a = gf(0xDEAD_BEEF, 0xCAFE_F00D);
        let b = gf(0x9E37_79B1_DEAD_BEEF, 0x1234_5678);
        let c = gf(0xA5A5_5A5A_F00D_BAAD, 0xDEAD_BEEF_CAFE_F00D);
        assert_eq!(a * b, b * a);
        assert_eq!((a * b) * c, a * (b * c));
    }

    #[test]
    fn distributivity_holds() {
        let a = gf(0xA5A5_A5A5_A5A5_A5A5, 0x5A5A_5A5A_5A5A_5A5A);
        let b = gf(0xDEAD_BEEF, 0xCAFE_F00D);
        let c = gf(0x1234_5678, 0x9ABC_DEF0);
        assert_eq!(a * (b + c), a * b + a * c);
    }

    #[test]
    fn inverse_satisfies_a_times_inv_equals_one() {
        let mut rng = StdRng::seed_from_u64(0xF00D_128);
        let one = BinaryFieldGF128::one();
        for _ in 0..64 {
            let mut a = rand_elt(&mut rng);
            if a.is_zero() {
                a = BinaryFieldGF128::one();
            }
            let inv = a.inverse();
            assert_eq!(a * inv, one);
        }
    }

    #[test]
    fn frobenius_squaring_is_linear() {
        // In characteristic 2 the Frobenius `x → x²` is additive:
        // `(x + y)² = x² + y²`.
        let mut rng = StdRng::seed_from_u64(0xBAAD_128);
        for _ in 0..256 {
            let a = rand_elt(&mut rng);
            let b = rand_elt(&mut rng);
            assert_eq!((a + b).square(), a.square() + b.square());
        }
    }

    #[test]
    fn karatsuba_clmul_matches_scalar_clmul() {
        // The Karatsuba `clmul_128x128` composes three 64×64 clmuls;
        // cross-check against a naive 4-mul layout via the scalar path,
        // so the Karatsuba bookkeeping (the XOR mixes) is right.
        let mut rng = StdRng::seed_from_u64(0xABCD_128);
        for _ in 0..64 {
            let a: [u64; 2] = [rng.random(), rng.random()];
            let b: [u64; 2] = [rng.random(), rng.random()];
            let karatsuba = clmul_128x128(&a, &b);

            let m00 = clmul_64x64_scalar(a[0], b[0]);
            let m01 = clmul_64x64_scalar(a[0], b[1]);
            let m10 = clmul_64x64_scalar(a[1], b[0]);
            let m11 = clmul_64x64_scalar(a[1], b[1]);
            let naive = [
                m00[0],
                m00[1] ^ m01[0] ^ m10[0],
                m11[0] ^ m01[1] ^ m10[1],
                m11[1],
            ];
            assert_eq!(karatsuba, naive);
        }
    }

    /// Frobenius: `a^{2^128} = a` for every `a ∈ GF(2^128)`. Computing
    /// 128 squarings is cheap.
    #[test]
    fn frobenius_equals_2_pow_128() {
        let cases = [
            gf(0xDEAD_BEEF, 0xCAFE_F00D),
            gf(0x9E37_79B1_DEAD_BEEF, 0x1234_5678),
            BinaryFieldGF128::one(),
        ];
        for a in cases {
            let mut x = a;
            for _ in 0..128 {
                x = x.square();
            }
            assert_eq!(x, a, "Frobenius failed: a^{{2^128}} should be a; got {x} for a = {a}");
        }
    }

    #[test]
    fn eval_f2_poly_at_one_returns_xor_of_bits() {
        // At α = 1, X^i = 1 for every i, so f(1) = popcount(f) mod 2.
        use crypto_primitives::boolean::Boolean;
        let bits5 = [0, 1, 3, 7, 31];
        let coeffs5: Vec<Boolean> = (0..32u32).map(|i| bits5.contains(&i).into()).collect();
        let p5: BinaryPoly<32> = BinaryPoly::new(coeffs5);
        let r5 = eval_f2_poly_d32_at(&p5, &BinaryFieldGF128::one());
        assert!(r5.is_one(), "eval at 1 of 5-bit poly should be 1; got {r5}");

        let bits4 = [0, 1, 3, 7];
        let coeffs4: Vec<Boolean> = (0..32u32).map(|i| bits4.contains(&i).into()).collect();
        let p4: BinaryPoly<32> = BinaryPoly::new(coeffs4);
        let r4 = eval_f2_poly_d32_at(&p4, &BinaryFieldGF128::one());
        assert!(r4.is_zero(), "eval at 1 of 4-bit poly should be 0; got {r4}");
    }

    #[test]
    fn eval_is_linear_over_f2() {
        let alpha = gf(0xDEAD_BEEF_CAFE_F00D, 0xA5A5);
        let p1_bits: u32 = 0xA5A5_A5A5;
        let p2_bits: u32 = 0xDEAD_BEEF;
        let p1: BinaryPoly<32> = BinaryPoly::from(p1_bits);
        let p2: BinaryPoly<32> = BinaryPoly::from(p2_bits);
        let sum: BinaryPoly<32> = BinaryPoly::from(p1_bits ^ p2_bits);

        let e1 = eval_f2_poly_d32_at(&p1, &alpha);
        let e2 = eval_f2_poly_d32_at(&p2, &alpha);
        let esum = eval_f2_poly_d32_at(&sum, &alpha);
        assert_eq!(esum, e1 + e2, "eval(p1 + p2) should equal eval(p1) + eval(p2)");
    }

    #[test]
    fn eval_is_compatible_with_multiplication_by_x() {
        let alpha = gf(0x9E37_79B1, 0xDEAD_BEEF);
        let p_bits: u32 = 0x1234_5678 & 0x7FFF_FFFF; // top bit clear
        let p: BinaryPoly<32> = BinaryPoly::from(p_bits);
        let xp: BinaryPoly<32> = BinaryPoly::from(p_bits << 1);
        let ep = eval_f2_poly_d32_at(&p, &alpha);
        let exp = eval_f2_poly_d32_at(&xp, &alpha);
        assert_eq!(exp, alpha * ep);
    }

    #[test]
    fn implements_field_and_prime_field_traits() {
        fn assert_field<F: crypto_primitives::Field>() {}
        fn assert_prime_field<F: crypto_primitives::PrimeField>() {}
        assert_field::<BinaryFieldGF128>();
        assert_prime_field::<BinaryFieldGF128>();
    }

    #[test]
    fn cfg_keyed_constructors_match_const_constructors() {
        use crypto_primitives::PrimeField;
        let cfg = ();
        assert_eq!(
            <BinaryFieldGF128 as PrimeField>::zero_with_cfg(&cfg),
            BinaryFieldGF128::zero(),
        );
        assert_eq!(
            <BinaryFieldGF128 as PrimeField>::one_with_cfg(&cfg),
            BinaryFieldGF128::one(),
        );
        let words = [0xDEAD_BEEFu64, 0xCAFEu64];
        let v: BinaryFieldGF128 = <BinaryFieldGF128 as PrimeField>::new_with_cfg(
            Uint::<2>::from_words(words),
            &cfg,
        );
        assert_eq!(*v.words(), words);
    }

    #[test]
    fn division_by_self_yields_one() {
        let a = gf(0xDEAD_BEEF, 0xCAFE_F00D);
        let b = a / a;
        assert!(b.is_one(), "a / a should be 1; got {b}");
    }

    #[test]
    #[should_panic(expected = "GF(2^128) has no prime modulus")]
    fn modulus_minus_one_div_two_panics() {
        use crypto_primitives::PrimeField;
        let a = BinaryFieldGF128::one();
        let _ = a.modulus_minus_one_div_two();
    }

    #[test]
    fn lift_then_eval_equals_direct_eval() {
        use crate::EvaluatablePolynomial;

        let alpha = gf(0xDEAD_BEEF_CAFE_F00D, 0xA5A5_BEEF);
        for bits in [0u32, 1, 0xA5A5_A5A5, 0xDEAD_BEEF, u32::MAX] {
            let p: BinaryPoly<32> = BinaryPoly::from(bits);
            let direct = eval_f2_poly_d32_at(&p, &alpha);
            let lifted: GF128Poly<32> = lift_f2_poly_to_gf128(&p);
            let horner = lifted.evaluate_at_point(&alpha).unwrap();
            assert_eq!(
                direct, horner,
                "lift-then-eval should equal direct eval for bits = {bits:#x}",
            );
        }
    }

    #[test]
    fn lift_is_linear_over_f2() {
        let p1: BinaryPoly<32> = BinaryPoly::from(0xA5A5_A5A5u32);
        let p2: BinaryPoly<32> = BinaryPoly::from(0xDEAD_BEEFu32);
        let p_sum: BinaryPoly<32> = BinaryPoly::from(0xA5A5_A5A5u32 ^ 0xDEAD_BEEFu32);

        let l1: GF128Poly<32> = lift_f2_poly_to_gf128(&p1);
        let l2: GF128Poly<32> = lift_f2_poly_to_gf128(&p2);
        let l_sum: GF128Poly<32> = lift_f2_poly_to_gf128(&p_sum);

        let sum_of_lifts = l1 + l2;
        assert_eq!(sum_of_lifts, l_sum);
    }

    #[test]
    fn lift_u64_and_lift_general_agree() {
        let bits: u32 = 0xCAFE_F00D;
        let p_general: BinaryPoly<32> = BinaryPoly::from(bits);
        let p_u64: BinaryU64Poly<32> = BinaryU64Poly::<32>::from(bits);
        let l_general: GF128Poly<32> = lift_f2_poly_to_gf128(&p_general);
        let l_u64: GF128Poly<32> = lift_f2_u64_poly_to_gf128(&p_u64);
        assert_eq!(l_general, l_u64);
    }

    #[test]
    fn lift_coefficients_are_zero_or_one() {
        let p: BinaryPoly<32> = BinaryPoly::from(0xDEAD_BEEFu32);
        let lifted: GF128Poly<32> = lift_f2_poly_to_gf128(&p);
        for c in lifted.coeffs.iter() {
            assert!(
                c.is_zero() || c.is_one(),
                "lifted coefficient must be 0 or 1 in GF(2^128); got {c}",
            );
        }
    }

    #[test]
    fn wide_eval_matches_d32_eval_for_low_degree() {
        let alpha = gf(0xA5A5, 0x5A5A);
        let bits: u32 = 0xDEAD_BEEF;
        let p: BinaryPoly<32> = BinaryPoly::from(bits);
        let pw: BinaryF2Poly<1> = BinaryF2Poly::from_words([bits as u64]);
        let e_d32 = eval_f2_poly_d32_at(&p, &alpha);
        let e_wide = eval_f2_wide_poly_at(&pw, &alpha);
        assert_eq!(e_d32, e_wide);
    }

    #[test]
    fn lift_and_project_are_bit_identical() {
        let g = gf(0x0123_4567_89AB_CDEF, 0xFEDC_BA98_7654_3210);
        let lifted = lift_gf128_to_f2_poly_2(&g);
        assert_eq!(lifted.words(), g.words());
        let back = f2_poly_2_to_gf128(&lifted);
        assert_eq!(back, g);
    }

    #[test]
    fn lift_and_project_zero_and_one() {
        let zero = BinaryFieldGF128::zero();
        let one = BinaryFieldGF128::one();
        assert_eq!(f2_poly_2_to_gf128(&lift_gf128_to_f2_poly_2(&zero)), zero);
        assert_eq!(f2_poly_2_to_gf128(&lift_gf128_to_f2_poly_2(&one)), one);
    }

    #[test]
    fn inverse_alpha_lift_round_trips() {
        // α is a transcript-fresh-style value (high entropy across
        // all 2 limbs). For any g ∈ GF(2^128), the inverse lift
        // must satisfy ψ_α(lift(g)) = g.
        let alpha = gf(0x0123_4567_89AB_CDEF, 0xFEDC_BA98_7654_3210);
        let basis = AlphaPolyBasis::new(&alpha);
        for g in [
            BinaryFieldGF128::zero(),
            BinaryFieldGF128::one(),
            gf(0x1, 0x0),
            gf(0xAAAA_BBBB_CCCC_DDDD, 0x1122_3344_5566_7788),
            alpha,
        ] {
            let lifted = basis.lift(&g);
            let recovered = eval_f2_wide_poly_at(&lifted, &alpha);
            assert_eq!(
                recovered, g,
                "inverse lift round-trip failed for g = {g:?}",
            );
        }
    }

    #[test]
    fn inverse_alpha_lift_is_f2_linear() {
        // ψ_α is F_2-linear, so the inverse lift must also be
        // F_2-linear: lift(a + b) = lift(a) + lift(b).
        let alpha = gf(0xCAFE_BABE, 0xDEAD_BEEF);
        let basis = AlphaPolyBasis::new(&alpha);
        let a = gf(0x1111_2222_3333_4444, 0x5555_6666_7777_8888);
        let b = gf(0xFEDC_BA98_7654_3210, 0xAAAA_BBBB_CCCC_DDDD);
        let sum = gf(
            0x1111_2222_3333_4444 ^ 0xFEDC_BA98_7654_3210,
            0x5555_6666_7777_8888 ^ 0xAAAA_BBBB_CCCC_DDDD,
        );
        let la = basis.lift(&a);
        let lb = basis.lift(&b);
        let lsum = basis.lift(&sum);
        let la_plus_lb = la + lb;
        assert_eq!(lsum, la_plus_lb);
    }

    #[test]
    fn psi_alpha_on_lifted_matches_eval_bits_at_alpha() {
        // For any g ∈ GF(2^128) viewed as a polynomial `Σ g_i · X^i`,
        // ψ_α(lift(g)) = Σ g_i · α^i.
        let alpha = gf(0x1, 0x2);
        let g = gf(0xAAAA_5555_AAAA_5555, 0x1234_5678_9ABC_DEF0);
        let lifted = lift_gf128_to_f2_poly_2(&g);
        let via_lift = eval_f2_wide_poly_at(&lifted, &alpha);

        // Reference: walk g's 128 bits, add α^i for each set bit.
        let mut ref_acc = BinaryFieldGF128::zero();
        let mut pow = BinaryFieldGF128::one();
        for word_idx in 0..2 {
            let mut w = g.words()[word_idx];
            let mut bit_in_word = 0usize;
            while bit_in_word < 64 {
                let _ = word_idx * 64 + bit_in_word;
                if (w & 1) == 1 {
                    ref_acc += &pow;
                }
                w >>= 1;
                bit_in_word += 1;
                if word_idx * 64 + bit_in_word < 128 {
                    pow *= &alpha;
                }
            }
        }
        assert_eq!(via_lift, ref_acc);
    }

    #[test]
    fn eval_with_powers_matches_branchless() {
        const D: usize = 32;
        let alpha = gf(
            0x9E37_79B9_7F4A_7C15,
            0xF39C_C060_5CEDC835,
        );
        let pows = alpha_powers(&alpha, D);

        let patterns: &[u64] = &[
            0,
            0xFFFF_FFFF,
            0x0000_0001,
            0x8000_0000,
            0xAAAA_AAAA,
            0x5555_5555,
            0xDEAD_BEEF,
            0x1234_5678,
            0xCAFE_F00D,
        ];
        for &bits in patterns {
            let cell = BinaryU64Poly::<D>::unpack_u64(bits);
            let optimized = eval_f2_poly_d_at_with_powers::<D>(&cell, &pows);
            let reference =
                eval_f2_poly_d_at_with_powers_branchless::<D>(&cell, &pows);
            assert_eq!(
                optimized, reference,
                "eval mismatch for bit pattern {bits:#x}"
            );
        }
    }

    /// SIMD-batched 4-cell projection must match the branchy single-cell
    /// kernel on every cell of every batch — that's the contract the
    /// prove-path swap depends on.
    #[test]
    fn simd_x4_matches_scalar_branchy_random() {
        use rand::{Rng, SeedableRng, rngs::StdRng};
        const D: usize = 32;
        let mut rng = StdRng::seed_from_u64(0x5_BA7CC0_128);
        let alpha = rand_elt(&mut rng);
        let pows = alpha_powers(&alpha, D);

        // 64 random batches × 4 cells = 256 single-cell checks plus the
        // SIMD bookkeeping check (correct cell ↔ result mapping).
        for _ in 0..64 {
            let cells_u64: [u64; 4] = [
                rng.random::<u32>() as u64,
                rng.random::<u32>() as u64,
                rng.random::<u32>() as u64,
                rng.random::<u32>() as u64,
            ];
            let simd = eval_f2_poly_d_at_with_powers_simd_x4::<D>(cells_u64, &pows);
            for k in 0..4 {
                let cell = BinaryU64Poly::<D>::unpack_u64(cells_u64[k]);
                let scalar = eval_f2_poly_d_at_with_powers::<D>(&cell, &pows);
                assert_eq!(
                    simd[k], scalar,
                    "SIMD lane {k} mismatch on bits[{:08x}, {:08x}, {:08x}, {:08x}]",
                    cells_u64[0], cells_u64[1], cells_u64[2], cells_u64[3],
                );
            }
        }
    }

    /// Edge-case patterns: all-zero, all-one, single bit at each
    /// position (validates the per-bit mask construction for every i).
    #[test]
    fn simd_x4_matches_scalar_branchy_edges() {
        const D: usize = 32;
        let alpha = gf(0x9E37_79B9_7F4A_7C15, 0xF39C_C060_5CED_C835);
        let pows = alpha_powers(&alpha, D);

        let mut patterns: Vec<u64> = vec![0, 0xFFFF_FFFFu32 as u64];
        for i in 0..D {
            patterns.push(1u64 << i);
        }
        // Window patterns: contiguous bit-runs at varying positions.
        patterns.extend([0x000F_F000, 0xFF00_FF00, 0x55AA_55AA, 0xDEAD_BEEF]);

        // Process patterns in groups of 4 and cross-check each lane.
        for chunk in patterns.chunks(4) {
            let mut cells_u64 = [0u64; 4];
            for (i, &b) in chunk.iter().enumerate() {
                cells_u64[i] = b;
            }
            let simd = eval_f2_poly_d_at_with_powers_simd_x4::<D>(cells_u64, &pows);
            for k in 0..chunk.len() {
                let cell = BinaryU64Poly::<D>::unpack_u64(cells_u64[k]);
                let scalar = eval_f2_poly_d_at_with_powers::<D>(&cell, &pows);
                assert_eq!(
                    simd[k], scalar,
                    "SIMD lane {k} edge-case mismatch on bits {:#x}",
                    cells_u64[k],
                );
            }
        }
    }

    /// Sparse-skip SIMD-x4 must match the dense SIMD-x4 (and therefore
    /// the scalar branchy kernel) on every batch.
    #[test]
    fn simd_x4_sparse_matches_simd_x4() {
        use rand::{Rng, SeedableRng, rngs::StdRng};
        const D: usize = 32;
        let mut rng = StdRng::seed_from_u64(0x5_AA70BA_128);
        let alpha = rand_elt(&mut rng);
        let pows = alpha_powers(&alpha, D);

        // Mix of random batches and adversarial cases: all-zero, all-one,
        // single-bit-only-in-one-cell.
        let mut batches: Vec<[u64; 4]> = vec![
            [0, 0, 0, 0],
            [0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF],
            [0x1, 0, 0, 0],           // only cell 0 has bit 0
            [0, 0x8000_0000, 0, 0],   // only cell 1 has bit 31
        ];
        for _ in 0..64 {
            batches.push([
                rng.random::<u32>() as u64,
                rng.random::<u32>() as u64,
                rng.random::<u32>() as u64,
                rng.random::<u32>() as u64,
            ]);
        }
        for cells in batches {
            let dense = eval_f2_poly_d_at_with_powers_simd_x4::<D>(cells, &pows);
            let sparse = eval_f2_poly_d_at_with_powers_simd_x4_sparse::<D>(cells, &pows);
            assert_eq!(
                dense, sparse,
                "sparse mismatch on bits[{:08x}, {:08x}, {:08x}, {:08x}]",
                cells[0], cells[1], cells[2], cells[3],
            );
        }
    }

    /// `project_column_with_powers` (chunks-of-4 + scalar remainder)
    /// must match a pure-scalar per-cell loop bit-for-bit, regardless
    /// of column length (covers `len % 4 = 0..3`).
    #[test]
    fn project_column_matches_scalar_per_cell() {
        use rand::{Rng, SeedableRng, rngs::StdRng};
        const D: usize = 32;
        let mut rng = StdRng::seed_from_u64(0xC0_1B_07_128);
        let alpha = rand_elt(&mut rng);
        let pows = alpha_powers(&alpha, D);

        // Column lengths spanning every (len % 4) remainder.
        for len in [0usize, 1, 2, 3, 4, 7, 8, 15, 16, 17, 33] {
            let cells: Vec<BinaryPoly<D>> = (0..len)
                .map(|_| BinaryU64Poly::<D>::unpack_u64(rng.random::<u32>() as u64))
                .collect();

            let batched = project_column_with_powers::<D>(&cells, &pows);
            let scalar: Vec<BinaryFieldGF128> = cells
                .iter()
                .map(|c| eval_f2_poly_d_at_with_powers::<D>(c, &pows))
                .collect();
            assert_eq!(batched, scalar, "column projection mismatch at len={len}");
        }
    }

    /// Scalar reference for `ic_accumulate_into_coeffs_bf128`. Mirrors
    /// the inner loop of `F2NativeIc::prove_linear` up_evals (modulo
    /// row-windowing, which the caller is responsible for).
    fn ic_accumulate_reference<const D: usize>(
        cells: &[BinaryPoly<D>],
        eq_table: &[BinaryFieldGF128],
        coeffs: &mut [BinaryFieldGF128],
    ) {
        assert_eq!(cells.len(), eq_table.len());
        assert!(coeffs.len() >= D);
        for i in 0..cells.len() {
            let mut bits = cells[i].pack_u64();
            if bits == 0 {
                continue;
            }
            let eq = eq_table[i];
            while bits != 0 {
                let d = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                coeffs[d] += eq;
            }
        }
    }

    #[test]
    fn ic_accumulate_bf128_matches_scalar() {
        const D: usize = 32;
        let mut rng = StdRng::seed_from_u64(0x1C_AC_C5_128);

        for &n in &[0usize, 1, 7, 64, 1024, 4097] {
            let cells: Vec<BinaryPoly<D>> = (0..n)
                .map(|_| BinaryU64Poly::<D>::unpack_u64(rng.random::<u32>() as u64))
                .collect();
            let eq: Vec<BinaryFieldGF128> = (0..n).map(|_| rand_elt(&mut rng)).collect();

            let mut got = vec![BinaryFieldGF128::zero(); D];
            ic_accumulate_into_coeffs_bf128::<D>(&cells, &eq, &mut got);

            let mut want = vec![BinaryFieldGF128::zero(); D];
            ic_accumulate_reference::<D>(&cells, &eq, &mut want);

            assert_eq!(got, want, "ic_accumulate mismatch at n={n}");
        }
    }

    #[test]
    fn ic_accumulate_4cols_simd_x4_matches_scalar() {
        const D: usize = 32;
        let mut rng = StdRng::seed_from_u64(0x4C_01_55_128);

        for &n in &[0usize, 1, 8, 1024, 8193] {
            let mk_cells = |rng: &mut StdRng| -> Vec<BinaryPoly<D>> {
                (0..n)
                    .map(|_| BinaryU64Poly::<D>::unpack_u64(rng.random::<u32>() as u64))
                    .collect()
            };
            let cells_a = mk_cells(&mut rng);
            let cells_b = mk_cells(&mut rng);
            let cells_c = mk_cells(&mut rng);
            let cells_d = mk_cells(&mut rng);
            let eq: Vec<BinaryFieldGF128> = (0..n).map(|_| rand_elt(&mut rng)).collect();

            let mut got_a = vec![BinaryFieldGF128::zero(); D];
            let mut got_b = vec![BinaryFieldGF128::zero(); D];
            let mut got_c = vec![BinaryFieldGF128::zero(); D];
            let mut got_d = vec![BinaryFieldGF128::zero(); D];
            ic_accumulate_4cols_simd_x4_bf128::<D>(
                &cells_a, &cells_b, &cells_c, &cells_d,
                &eq,
                &mut got_a, &mut got_b, &mut got_c, &mut got_d,
            );

            let mut want_a = vec![BinaryFieldGF128::zero(); D];
            let mut want_b = vec![BinaryFieldGF128::zero(); D];
            let mut want_c = vec![BinaryFieldGF128::zero(); D];
            let mut want_d = vec![BinaryFieldGF128::zero(); D];
            ic_accumulate_reference::<D>(&cells_a, &eq, &mut want_a);
            ic_accumulate_reference::<D>(&cells_b, &eq, &mut want_b);
            ic_accumulate_reference::<D>(&cells_c, &eq, &mut want_c);
            ic_accumulate_reference::<D>(&cells_d, &eq, &mut want_d);

            assert_eq!(got_a, want_a, "col A mismatch at n={n}");
            assert_eq!(got_b, want_b, "col B mismatch at n={n}");
            assert_eq!(got_c, want_c, "col C mismatch at n={n}");
            assert_eq!(got_d, want_d, "col D mismatch at n={n}");
        }
    }
}
