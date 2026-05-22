//! `GF(2^192)`: degree-192 binary extension field.
//!
//! Elements are `F_2[X] / <f(X)>` where `f(X)` is the NIST FIPS 186-2
//! pentanomial reduction polynomial
//!
//! ```text
//! f(X) = X^192 + X^7 + X^2 + X + 1.
//! ```
//!
//! `f` is irreducible over `F_2`. The factor ring is therefore a field
//! of order `2^192`. Its multiplicative group has order `2^192 - 1`,
//! so for every nonzero `a` we have `a^{2^192 - 1} = 1` and
//! `a^{-1} = a^{2^192 - 2}`.
//!
//! Storage layout: each element is a `[u64; 3]` bit-packed polynomial
//! of degree `< 192`. Bit `i` of word `i/64` holds the coefficient of
//! `X^i`. Addition is XOR; multiplication is the F_2 carryless product
//! followed by reduction modulo `f`.
//!
//! Intended use: the random "projecting element" `α` in the F_2 proving
//! path. After the ideal check runs over `F_2[X]`, the protocol samples
//! `α ∈ GF(2^192)` and substitutes `X = α` in every committed cell, so
//! that the sumcheck-based phase runs over `GF(2^192)` instead of a
//! prime field.

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
    binary::BinaryPoly, binary_f2_wide::BinaryF2Poly, binary_u64::BinaryU64Poly,
    dense::DensePolynomial,
};

/// A `GF(2^192)[X]<D>` polynomial — degree-`<D` univariate with
/// `GF(2^192)`-valued coefficients. The natural target of the
/// `F_2[X] → GF(2^192)[X]` coefficient lift used in step 2 of the
/// F_2 proving path (see `protocol/src/f2_prove_plan.md`).
///
/// Built on top of the existing [`DensePolynomial`] machinery:
/// `BinaryFieldGF192` already implements `Semiring` (via the
/// degenerate `PrimeField` impl), so addition / negation /
/// `EvaluatablePolynomial<R, R>` (Horner at a point) all come
/// for free.
pub type GF192Poly<const D: usize> = DensePolynomial<BinaryFieldGF192, D>;

/// Low bits of the reduction polynomial — `g(X) = X^7 + X^2 + X + 1`,
/// stored as the bit pattern `0b 1000_0111 = 0x87`. (The `X^192` term
/// is implicit in the reduction map below.)
pub const REDUCTION_LOW: u64 = 0x87;

/// `Field::Modulus`-shaped representation of the reduction polynomial:
/// the same low-bit pattern as [`REDUCTION_LOW`], promoted to a
/// `Uint<3>` so the byte width matches `Field::Inner` (a requirement
/// of `Transcript::absorb_random_field`).
pub const MODULUS_LOW_BITS: Uint<3> = Uint::<3>::from_words([REDUCTION_LOW, 0, 0]);

/// An element of `GF(2^192) = F_2[X] / <X^192 + X^7 + X^2 + X + 1>`.
///
/// Stored as a [`Uint<3>`] (3 × `u64` = 192 bits). Bit `64*w + b`
/// (LSB-first within each `u64` limb) holds the coefficient of
/// `X^{64*w + b}`; bits `0..192` carry the value, no bits at or
/// above 192 are ever set in a reduced element.
///
/// The choice of `Uint<3>` for storage is load-bearing for the
/// transcript / `Field::Inner` plumbing: `Uint<L>` already
/// implements `ConstTranscribable`, which is what the IC's
/// `transcript.get_field_challenge::<F>` API requires of
/// `F::Inner`.
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
#[repr(transparent)]
pub struct BinaryFieldGF192 {
    uint: Uint<3>,
}

impl BinaryFieldGF192 {
    pub const fn zero() -> Self {
        Self { uint: Uint::<3>::ZERO }
    }

    pub const fn one() -> Self {
        Self {
            uint: Uint::<3>::from_words([1u64, 0, 0]),
        }
    }

    /// Construct from raw bit-packed words. Caller is responsible for
    /// ensuring the value is already reduced (no bits ≥ 192 set; here
    /// that's the natural invariant since the array only has 192 bits).
    pub const fn from_words(words: [u64; 3]) -> Self {
        Self {
            uint: Uint::<3>::from_words(words),
        }
    }

    /// Borrow the bit-packed representation. Bit `i` of word `i / 64`
    /// holds the coefficient of `X^i`.
    pub const fn words(&self) -> &[u64; 3] {
        self.uint.as_words()
    }

    pub fn is_zero(&self) -> bool {
        let w = self.uint.as_words();
        w[0] == 0 && w[1] == 0 && w[2] == 0
    }

    /// `a^2` — Frobenius squaring. Composite of carryless square + reduce.
    pub fn square(&self) -> Self {
        let product = clmul_192x192(self.uint.as_words(), self.uint.as_words());
        Self::from_words(reduce_384_to_192(product))
    }

    /// Inverse via Fermat: `a^{-1} = a^{2^192 - 2}` (for `a ≠ 0`).
    ///
    /// `2^192 - 2 = 2 · (2^191 - 1)`, so we compute `b = a^{2^191 - 1}`
    /// and return `b^2`. `b` is built via the standard "all-ones
    /// exponent" loop: if `c_k = a^{2^k - 1}` then
    /// `c_{k+1} = c_k^2 · a`. After 190 such steps from `c_1 = a`,
    /// `c_191 = a^{2^191 - 1}`. Total cost: 191 squarings + 190 mults.
    ///
    /// This is the naive chain — an addition-chain optimised inverse
    /// (`c_{2k} = c_k^{2^k} · c_k`) would cut multiplications to ~10
    /// and squarings to 192. Doable later if profiling demands it.
    ///
    /// Panics if `self.is_zero()` — `0` has no multiplicative inverse.
    pub fn inverse(&self) -> Self {
        assert!(!self.is_zero(), "GF(2^192): zero has no inverse");
        let mut c = *self; // c = a^{2^1 - 1} = a
        for _ in 1..191 {
            c = c.square();
            c *= self;
        }
        // c = a^{2^191 - 1}. One more squaring → a^{2 · (2^191 - 1)} = a^{2^192 - 2}.
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


impl Zero for BinaryFieldGF192 {
    fn zero() -> Self {
        Self::zero()
    }
    fn is_zero(&self) -> bool {
        Self::is_zero(self)
    }
}

impl One for BinaryFieldGF192 {
    fn one() -> Self {
        Self::one()
    }
    fn is_one(&self) -> bool {
        let w = self.uint.as_words();
        w[0] == 1 && w[1] == 0 && w[2] == 0
    }
}

impl Display for BinaryFieldGF192 {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let w = self.uint.as_words();
        write!(f, "GF192[{:016x}_{:016x}_{:016x}]", w[2], w[1], w[0])
    }
}

// -- additive group --------------------------------------------------

impl<'a> AddAssign<&'a Self> for BinaryFieldGF192 {
    #[inline]
    #[allow(clippy::arithmetic_side_effects, clippy::suspicious_op_assign_impl)]
    fn add_assign(&mut self, rhs: &'a Self) {
        // `Uint<3>` doesn't directly expose word-level mutation, so we
        // read both operand word arrays out, XOR, and rebuild.
        let lw = *self.uint.as_words();
        let rw = rhs.uint.as_words();
        self.uint = Uint::<3>::from_words([lw[0] ^ rw[0], lw[1] ^ rw[1], lw[2] ^ rw[2]]);
    }
}

impl AddAssign<Self> for BinaryFieldGF192 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        <Self as AddAssign<&Self>>::add_assign(self, &rhs);
    }
}

impl<'a> Add<&'a Self> for BinaryFieldGF192 {
    type Output = Self;
    #[inline]
    fn add(mut self, rhs: &'a Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl Add<Self> for BinaryFieldGF192 {
    type Output = Self;
    #[inline]
    fn add(mut self, rhs: Self) -> Self::Output {
        self += &rhs;
        self
    }
}

// In characteristic 2, subtraction == addition (XOR).
impl<'a> SubAssign<&'a Self> for BinaryFieldGF192 {
    #[inline]
    fn sub_assign(&mut self, rhs: &'a Self) {
        <Self as AddAssign<&Self>>::add_assign(self, rhs);
    }
}
impl SubAssign<Self> for BinaryFieldGF192 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self += rhs;
    }
}
impl<'a> Sub<&'a Self> for BinaryFieldGF192 {
    type Output = Self;
    #[inline]
    fn sub(mut self, rhs: &'a Self) -> Self::Output {
        self -= rhs;
        self
    }
}
impl Sub<Self> for BinaryFieldGF192 {
    type Output = Self;
    #[inline]
    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= &rhs;
        self
    }
}

impl Neg for BinaryFieldGF192 {
    type Output = Self;
    /// In characteristic 2, every element is its own additive inverse.
    #[inline]
    fn neg(self) -> Self::Output {
        self
    }
}

// -- multiplicative group --------------------------------------------

impl<'a> MulAssign<&'a Self> for BinaryFieldGF192 {
    #[inline]
    fn mul_assign(&mut self, rhs: &'a Self) {
        let product = clmul_192x192(self.uint.as_words(), rhs.uint.as_words());
        self.uint = Uint::<3>::from_words(reduce_384_to_192(product));
    }
}
impl MulAssign<Self> for BinaryFieldGF192 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        <Self as MulAssign<&Self>>::mul_assign(self, &rhs);
    }
}
impl<'a> Mul<&'a Self> for BinaryFieldGF192 {
    type Output = Self;
    #[inline]
    fn mul(mut self, rhs: &'a Self) -> Self::Output {
        self *= rhs;
        self
    }
}
impl Mul<Self> for BinaryFieldGF192 {
    type Output = Self;
    #[inline]
    fn mul(mut self, rhs: Self) -> Self::Output {
        self *= &rhs;
        self
    }
}

// -- division (multiply by inverse) ----------------------------------

impl<'a> DivAssign<&'a Self> for BinaryFieldGF192 {
    /// `self /= rhs` ≡ `self *= rhs.inverse()`. Panics if `rhs` is zero.
    #[inline]
    fn div_assign(&mut self, rhs: &'a Self) {
        *self *= rhs.inverse();
    }
}
impl DivAssign<Self> for BinaryFieldGF192 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self /= &rhs;
    }
}
impl<'a> Div<&'a Self> for BinaryFieldGF192 {
    type Output = Self;
    #[inline]
    fn div(mut self, rhs: &'a Self) -> Self::Output {
        self /= rhs;
        self
    }
}
impl Div<Self> for BinaryFieldGF192 {
    type Output = Self;
    #[inline]
    fn div(mut self, rhs: Self) -> Self::Output {
        self /= &rhs;
        self
    }
}

// -- num_traits checked-* (degenerate in a field — never fail) -------
//
// `Semiring` requires `CheckedAdd + CheckedSub + CheckedMul` and
// `Ring` adds `CheckedNeg`. In characteristic 2 there is no integer-
// style overflow at any of these, so every checked op returns `Some`.

impl CheckedAdd for BinaryFieldGF192 {
    #[inline]
    fn checked_add(&self, rhs: &Self) -> Option<Self> {
        Some(*self + rhs)
    }
}
impl CheckedSub for BinaryFieldGF192 {
    #[inline]
    fn checked_sub(&self, rhs: &Self) -> Option<Self> {
        Some(*self - rhs)
    }
}
impl CheckedMul for BinaryFieldGF192 {
    #[inline]
    fn checked_mul(&self, rhs: &Self) -> Option<Self> {
        Some(*self * rhs)
    }
}
impl CheckedNeg for BinaryFieldGF192 {
    #[inline]
    fn checked_neg(&self) -> Option<Self> {
        Some(-*self)
    }
}

// -- Sum / Product folds ---------------------------------------------

impl Sum<Self> for BinaryFieldGF192 {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}
impl<'a> Sum<&'a Self> for BinaryFieldGF192 {
    #[inline]
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + *x)
    }
}
impl Product<Self> for BinaryFieldGF192 {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}
impl<'a> Product<&'a Self> for BinaryFieldGF192 {
    #[inline]
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * *x)
    }
}

// -- Inv ------------------------------------------------------------

impl Inv for BinaryFieldGF192 {
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

impl Pow<u32> for BinaryFieldGF192 {
    type Output = Self;
    #[inline]
    fn pow(self, exp: u32) -> Self::Output {
        self.pow_u32(exp)
    }
}

// -- ConstZero / ConstOne / From<bool> ------------------------------

impl ConstZero for BinaryFieldGF192 {
    const ZERO: Self = Self::zero();
}

impl ConstOne for BinaryFieldGF192 {
    const ONE: Self = Self::one();
}

impl From<bool> for BinaryFieldGF192 {
    #[inline]
    fn from(b: bool) -> Self {
        if b { Self::one() } else { Self::zero() }
    }
}

// -- From<{u8..u128, i8..i128}> --------------------------------------
//
// `FromPrimitiveWithConfig` (used by `MultiDegreeSumcheck::prove_*`
// to inject `num_vars`, `degree`, etc. into the transcript) is the
// umbrella trait `FromWithConfig<u8> + ... + FromWithConfig<i128>`.
// `FromWithConfig` has a blanket impl over `PrimeField + From<T>`, so
// we get the whole bundle by providing the 10 `From` impls below.
//
// Semantics: interpret the primitive's bit pattern as the low
// coefficients of a GF(2^192) element — i.e., `from(n: u64) = n`
// stored as bit-pattern words `[n, 0, 0]`. This is the only
// deterministic, reversible map from primitives to GF(2^192) that
// makes sense in characteristic 2 (there's no integer modulus to
// reduce against). Signed types use their two's-complement bit
// pattern, sign-extended to `u128`. The transcript only uses this
// map for hash absorption / challenge derivation, so any
// deterministic injection works; it just needs to be the same on
// prover and verifier (which it trivially is, being purely
// arithmetic-free bit copies).

macro_rules! impl_from_unsigned {
    ($($t:ty),*) => {
        $(
            impl From<$t> for BinaryFieldGF192 {
                #[inline]
                fn from(v: $t) -> Self {
                    Self::from_words([v as u64, 0, 0])
                }
            }
        )*
    };
}

impl_from_unsigned!(u8, u16, u32, u64);

impl From<u128> for BinaryFieldGF192 {
    #[inline]
    fn from(v: u128) -> Self {
        Self::from_words([v as u64, (v >> 64) as u64, 0])
    }
}

macro_rules! impl_from_signed {
    ($($t:ty),*) => {
        $(
            impl From<$t> for BinaryFieldGF192 {
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

impl From<i128> for BinaryFieldGF192 {
    #[inline]
    fn from(v: i128) -> Self {
        Self::from(v as u128)
    }
}

// -- Semiring / Ring marker traits ----------------------------------

impl Semiring for BinaryFieldGF192 {}
impl Ring for BinaryFieldGF192 {}

// -- Field ----------------------------------------------------------

impl Field for BinaryFieldGF192 {
    /// Bit-packed `Uint<3>` (192 bits, LSB-first within each `u64` limb).
    /// We use `Uint<3>` rather than `[u64; 3]` so that
    /// `F::Inner: ConstTranscribable` is satisfied directly (the
    /// transcript / Fiat-Shamir API uses `Inner` as the byte-level
    /// challenge representation).
    type Inner = Uint<3>;
    /// The reduction polynomial is hardcoded
    /// (`X^192 + X^7 + X^2 + X + 1`). Stored as the **low 192 bits**
    /// of `f(X)` — the `X^192` term is implicit, so the representable
    /// pattern is just `g(X) = X^7 + X^2 + X + 1` = `0x87`.
    ///
    /// We use `Uint<3>` rather than `()` so that
    /// `F::Modulus` shares the byte length of `F::Inner`. The
    /// transcript's `absorb_random_field` interleaves `(modulus,
    /// inner)` bytes and asserts the two byte lengths match — `()`
    /// would break that invariant. The actual bit value is fixed; any
    /// non-`MODULUS_LOW_BITS` value handed to `make_cfg` is rejected.
    type Modulus = Uint<3>;

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
// `BinaryFieldGF192` is a binary extension field, NOT a prime field.
// The codebase's `PrimeField` abstraction predates the F_2 work and is
// the central bound throughout `piop/` and `protocol/`. To reuse that
// machinery without a wide refactor, we implement `PrimeField`
// degenerately:
//
// - `Config = ()`: the field has no runtime-configurable data (the
//   reduction polynomial is compile-time-fixed).
// - `Modulus = ()` (inherited from `Field`): same reason.
// - `modulus()` / `make_cfg()`: return `()` / `Ok(())`.
// - `modulus_minus_one_div_two()`: **panics**. This method has no
//   meaningful analogue in characteristic 2 (no integer modulus, no
//   notion of "half the modulus minus one"); it is used by prime-field
//   square-root / Legendre-symbol code paths that are nonsense for
//   `GF(2^192)`. A panic is preferable to a silent wrong answer if a
//   downstream call site reaches it.
//
// The semantic mismatch is documented at the call-site level: any
// generic protocol code parameterised over `F: PrimeField` will compile
// against `BinaryFieldGF192`, but `modulus_minus_one_div_two`-shaped
// paths must be audited before claiming soundness.

impl PrimeField for BinaryFieldGF192 {
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
        MODULUS_LOW_BITS
    }

    fn modulus_minus_one_div_two(&self) -> Self::Inner {
        panic!(
            "BinaryFieldGF192::modulus_minus_one_div_two: GF(2^192) has no \
             prime modulus; this method is degenerate. Audit the call site \
             before using it with a binary extension field."
        );
    }

    fn make_cfg(modulus: &Self::Modulus) -> Result<Self::Config, FieldError> {
        if modulus == &MODULUS_LOW_BITS {
            Ok(())
        } else {
            Err(FieldError::InvalidModulus)
        }
    }

    #[inline(always)]
    fn new_with_cfg(inner: Self::Inner, _cfg: &Self::Config) -> Self {
        // Input is a 192-bit `Uint<3>`; no high bits to reduce.
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
// `BinaryFieldGF192`, the inner repr IS the field-element repr
// (`Uint<3>` = bit-packed F_2[X]/<f>) — there's no Montgomery
// reinterpretation needed. Each method below just delegates to the
// regular F_2 field ops.

impl InnerTransparentField for BinaryFieldGF192 {
    #[inline]
    fn add_inner(
        lhs: &Self::Inner,
        rhs: &Self::Inner,
        _config: &Self::Config,
    ) -> Self::Inner {
        let lw = lhs.as_words();
        let rw = rhs.as_words();
        Uint::<3>::from_words([lw[0] ^ rw[0], lw[1] ^ rw[1], lw[2] ^ rw[2]])
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
        // `BinaryFieldGF192` and reuse the regular `MulAssign<&Self>`.
        let r = Self { uint: *rhs };
        *self *= &r;
    }
}

// -- carryless multiplication and reduction --------------------------

/// F_2 polynomial multiplication of two 192-bit operands → 384 bits.
/// Schoolbook bit-by-bit on the lhs; O(192 × 3) word ops.
#[allow(clippy::arithmetic_side_effects)]
fn clmul_192x192(a: &[u64; 3], b: &[u64; 3]) -> [u64; 6] {
    let mut acc = [0u64; 6];
    for ai in 0..3 {
        let mut a_word = a[ai];
        while a_word != 0 {
            let lo = a_word.trailing_zeros() as usize;
            let shift = 64 * ai + lo;
            xor_b_shifted(&mut acc, b, shift);
            a_word &= a_word - 1;
        }
    }
    acc
}

/// XOR `b << shift` (`b` is the 192-bit operand) into `acc` (384 bits).
#[inline]
#[allow(clippy::arithmetic_side_effects)]
fn xor_b_shifted(acc: &mut [u64; 6], b: &[u64; 3], shift: usize) {
    let word_shift = shift / 64;
    let bit_shift = shift % 64;
    for i in 0..3 {
        let bw = b[i];
        let dst = word_shift + i;
        if bit_shift == 0 {
            acc[dst] ^= bw;
        } else {
            acc[dst] ^= bw << bit_shift;
            acc[dst + 1] ^= bw >> (64 - bit_shift);
        }
    }
}

/// Reduce a 384-bit `F_2[X]` polynomial modulo
/// `f(X) = X^192 + X^7 + X^2 + X + 1`.
///
/// For each bit `i ≥ 192` in the input, `X^i ≡ X^{i-192} · g(X)`
/// (mod `f`), with `g(X) = X^7 + X^2 + X + 1`. Word-at-a-time: for
/// each high word `w` of the input, the contribution to the reduced
/// result is the 71-bit polynomial `w · g`, shifted into position.
#[allow(clippy::arithmetic_side_effects)]
fn reduce_384_to_192(prod: [u64; 6]) -> [u64; 3] {
    // Process high words from 5 down to 3 so the cascade from word 5
    // into word 3 (bits 192..198) gets re-reduced by step 3.
    let mut r = [prod[0], prod[1], prod[2]];
    let mut hi3 = prod[3]; // bits 192..255

    // Step A: reduce word 5 (bits 320..383). X^{320+k} ≡ X^{128+k} · g.
    // Result lands in r[2] (bits 128..191) and into bits 192..198,
    // which we accumulate into `hi3` for re-reduction in step C.
    let w = prod[5];
    let (lo, hi) = word_times_g(w);
    r[2] ^= lo;
    hi3 ^= hi; // hi has at most bits 0..6 set (positions 192..198 globally)

    // Step B: reduce word 4 (bits 256..319). X^{256+k} ≡ X^{64+k} · g.
    // Lands in r[1] (bits 64..127) and r[2] (bits 128..191).
    let w = prod[4];
    let (lo, hi) = word_times_g(w);
    r[1] ^= lo;
    r[2] ^= hi;

    // Step C: reduce updated `hi3` (bits 192..255 of input ⊕ overflow
    // from step A). X^{192+k} ≡ X^k · g; max bit position is
    // `63 + 7 = 70`, so the result lands in r[0] and r[1].
    let w = hi3;
    let (lo, hi) = word_times_g(w);
    r[0] ^= lo;
    r[1] ^= hi;

    r
}

/// `w · g(X)` where `g(X) = X^7 + X^2 + X + 1`. Returns `(lo, hi)` —
/// `lo` is bits 0..63 of the product, `hi` is bits 64..70 (degree ≤ 70
/// from the `<< 7` overflow); bits ≥ 71 of `hi` are zero.
#[inline]
#[allow(clippy::arithmetic_side_effects)]
fn word_times_g(w: u64) -> (u64, u64) {
    let lo = w ^ (w << 1) ^ (w << 2) ^ (w << 7);
    let hi = (w >> 63) ^ (w >> 62) ^ (w >> 57);
    (lo, hi)
}

// -- F_2[X]<D> → GF(2^192)[X]<D> coefficient lift --------------------
//
// Every `F_2` element is canonically a `GF(2^192)` element via the
// inclusion `F_2 ⊂ GF(2^192)`: `Boolean::ZERO → GF192::zero()` and
// `Boolean::ONE → GF192::one()`. Applied per-coefficient, this
// extends an `F_2[X]<D>`-typed cell to a `GF(2^192)[X]<D>` value
// the IC can then combine with `GF(2^192)`-valued challenges.
//
// The lift is the identity-on-cells composed with the trivial
// embedding of the coefficient ring — no randomness involved. By
// construction `eval_after_lift(p, α) = eval_f2(p, α)` (proved in
// `lift_then_eval_equals_direct_eval`).

/// Lift `F_2[X]<D>` (a [`BinaryPoly<D>`]) to `GF(2^192)[X]<D>`.
/// Each `Boolean` coefficient maps to `GF192::zero()` or
/// `GF192::one()`.
pub fn lift_f2_poly_to_gf192<const D: usize>(p: &BinaryPoly<D>) -> GF192Poly<D> {
    let mut coeffs: [BinaryFieldGF192; D] = [BinaryFieldGF192::zero(); D];
    for (i, c) in p.iter().enumerate() {
        if c.inner() {
            coeffs[i] = BinaryFieldGF192::one();
        }
    }
    DensePolynomial { coeffs }
}

/// Lift `F_2[X]<D>` (a [`BinaryU64Poly<D>`], `D ≤ 64`) to
/// `GF(2^192)[X]<D>`. Same per-coefficient embedding as
/// [`lift_f2_poly_to_gf192`], but reading from the bit-packed
/// representation directly.
pub fn lift_f2_u64_poly_to_gf192<const D: usize>(p: &BinaryU64Poly<D>) -> GF192Poly<D> {
    assert!(D <= 64, "lift_f2_u64_poly_to_gf192: D ({D}) must be ≤ 64");
    let bits = *p.inner();
    let mut coeffs: [BinaryFieldGF192; D] = [BinaryFieldGF192::zero(); D];
    for (i, slot) in coeffs.iter_mut().enumerate().take(D) {
        #[allow(clippy::arithmetic_side_effects)]
        let bit = (bits >> i) & 1;
        if bit == 1 {
            *slot = BinaryFieldGF192::one();
        }
    }
    DensePolynomial { coeffs }
}

// -- evaluation: F_2[X]<D> → GF(2^192) at X = α ----------------------

/// Substitute `X = alpha` in an `F_2[X]<D>`-typed cell. Each set bit
/// `i` of `p` adds `alpha^i` to the running accumulator. Computed via
/// `(α^0, α^1, α^2, …)` Horner-style: scan bits of `p`'s 32-bit-or-
/// smaller pattern and XOR in the current power of α whenever a bit
/// is set. Total cost: `D - 1` field squarings + `popcount(p)` adds.
pub fn eval_f2_poly_d32_at(p: &BinaryPoly<32>, alpha: &BinaryFieldGF192) -> BinaryFieldGF192 {
    let mut bits: u64 = 0;
    for (i, c) in p.iter().enumerate() {
        if c.inner() {
            #[allow(clippy::arithmetic_side_effects)]
            {
                bits |= 1u64 << i;
            }
        }
    }
    eval_bits_at(bits, 32, alpha)
}

/// Substitute `X = alpha` in an `F_2[X]<D>`-typed cell stored in
/// [`BinaryU64Poly`] form. `D` must be ≤ 64.
pub fn eval_f2_u64_poly_at<const D: usize>(
    p: &BinaryU64Poly<D>,
    alpha: &BinaryFieldGF192,
) -> BinaryFieldGF192 {
    assert!(D <= 64, "eval_f2_u64_poly_at: D ({D}) must be ≤ 64");
    eval_bits_at(*p.inner(), D, alpha)
}

/// Substitute `X = alpha` in a [`BinaryF2Poly<W>`] (wide F_2[X]
/// stored as `[u64; W]`).
#[allow(clippy::arithmetic_side_effects)]
pub fn eval_f2_wide_poly_at<const W: usize>(
    p: &BinaryF2Poly<W>,
    alpha: &BinaryFieldGF192,
) -> BinaryFieldGF192 {
    let mut acc = BinaryFieldGF192::zero();
    let mut pow = BinaryFieldGF192::one(); // α^0
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

/// Inner kernel for `eval_*` variants whose `<D>` polynomial fits in a
/// `u64`. Walks bits of `bits` from LSB up, multiplying a running
/// `pow = α^i` by `alpha` at each step.
#[inline]
#[allow(clippy::arithmetic_side_effects)]
fn eval_bits_at(mut bits: u64, max_bits: usize, alpha: &BinaryFieldGF192) -> BinaryFieldGF192 {
    debug_assert!(max_bits <= 64);
    let mut acc = BinaryFieldGF192::zero();
    let mut pow = BinaryFieldGF192::one(); // α^0
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

#[cfg(test)]
mod tests {
    use super::*;

    fn gf(lo: u64, mid: u64, hi: u64) -> BinaryFieldGF192 {
        BinaryFieldGF192::from_words([lo, mid, hi])
    }

    #[test]
    fn zero_and_one_are_identities() {
        let z = BinaryFieldGF192::zero();
        let o = BinaryFieldGF192::one();
        let a = gf(0xDEAD_BEEF_CAFE_F00D, 0xA5A5_A5A5_A5A5_A5A5, 0xDEAD_BEEF);
        assert_eq!(a + z, a);
        assert_eq!(a * o, a);
        assert_eq!(a * z, z);
    }

    #[test]
    fn addition_is_xor_and_self_inverse() {
        let a = gf(0xDEAD_BEEF, 0xCAFE_F00D, 0x12345);
        let b = gf(0x9E37_79B1, 0x1234_5678, 0x67890);
        let sum = a + b;
        let again = sum + b;
        assert_eq!(again, a, "char-2: (a + b) + b = a");
        assert_eq!(a + a, BinaryFieldGF192::zero(), "char-2: a + a = 0");
    }

    #[test]
    fn multiplication_is_commutative_and_associative() {
        let a = gf(0xDEAD_BEEF, 0xCAFE_F00D, 0x12345);
        let b = gf(0x9E37_79B1_DEAD_BEEF, 0x1234_5678, 0x6789_0ABC);
        let c = gf(0xA5A5_5A5A_F00D_BAAD, 0xDEAD_BEEF_CAFE_F00D, 0xFFFF);
        assert_eq!(a * b, b * a);
        assert_eq!((a * b) * c, a * (b * c));
    }

    #[test]
    fn distributivity_holds() {
        let a = gf(0xA5A5_A5A5_A5A5_A5A5, 0x5A5A_5A5A_5A5A_5A5A, 0xDEAD);
        let b = gf(0xDEAD_BEEF, 0xCAFE_F00D, 0x12345);
        let c = gf(0x1234_5678, 0x9ABC_DEF0, 0x55AA_55AA_55AA_55AA);
        assert_eq!(a * (b + c), a * b + a * c);
    }

    #[test]
    fn multiplication_stays_within_192_bits() {
        // Reduction: high bits ≥ 192 should always be zero after a
        // multiplication. Multiply two 192-bit values that are both
        // "near-maximum" to force reduction to engage.
        let a = gf(u64::MAX, u64::MAX, u64::MAX);
        let b = gf(u64::MAX, u64::MAX, u64::MAX);
        let prod = a * b;
        // Field elements only have 192 bits; the storage is `[u64; 3]`,
        // so there are no out-of-range bits to check — the operation
        // just needs to produce *some* 192-bit answer without panic.
        // The real correctness checks are in `frobenius_equals_2_pow_192`
        // and the inverse-roundtrip test.
        let _ = prod;
    }

    /// Inverse roundtrip: for many random nonzero `a`, `a · a^{-1} = 1`.
    #[test]
    fn inverse_roundtrip() {
        let cases = [
            gf(0xDEAD_BEEF_CAFE_F00D, 0xA5A5_A5A5_A5A5_A5A5, 0x12345),
            gf(0x0000_0000_0000_0001, 0x0000_0000_0000_0000, 0x0000_0000_0000_0000),
            gf(0x9E37_79B1_DEAD_BEEF, 0xCAFE_F00D_F00D_BAAD, 0xFFFF_FFFF_0000_0000),
            gf(0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF, 0x0000_FFFF_FFFF_FFFF),
            gf(0x1, 0x0, 0x0), // a = 1, inverse = 1
        ];
        for a in cases {
            let ainv = a.inverse();
            let prod = a * ainv;
            assert!(prod.is_one(), "a · a^-1 should be 1; got {prod} for a = {a}");
        }
    }

    /// Frobenius: `a^{2^192} = a` for every `a ∈ GF(2^192)`. Equivalently
    /// `a^{2^192 - 1} = 1` for nonzero `a`. Computing 192 squarings is
    /// cheap; we then check `a · ··· = a`.
    #[test]
    fn frobenius_equals_2_pow_192() {
        let cases = [
            gf(0xDEAD_BEEF, 0xCAFE_F00D, 0x12345),
            gf(0x9E37_79B1_DEAD_BEEF, 0x1234_5678, 0x6789_0ABC),
            BinaryFieldGF192::one(),
        ];
        for a in cases {
            let mut x = a;
            for _ in 0..192 {
                x = x.square();
            }
            assert_eq!(x, a, "Frobenius failed: a^{{2^192}} should be a; got {x} for a = {a}");
        }
    }

    #[test]
    fn eval_f2_poly_at_one_returns_xor_of_bits() {
        // At α = 1, X^i = 1 for every i, so f(1) = popcount(f) mod 2.
        // For a polynomial with 5 set bits, eval(1) should be 1.
        // For 4 set bits, eval(1) should be 0.
        use crypto_primitives::boolean::Boolean;
        let bits5 = [0, 1, 3, 7, 31];
        let coeffs5: Vec<Boolean> = (0..32u32).map(|i| bits5.contains(&i).into()).collect();
        let p5: BinaryPoly<32> = BinaryPoly::new(coeffs5);
        let r5 = eval_f2_poly_d32_at(&p5, &BinaryFieldGF192::one());
        assert!(r5.is_one(), "eval at 1 of 5-bit poly should be 1; got {r5}");

        let bits4 = [0, 1, 3, 7];
        let coeffs4: Vec<Boolean> = (0..32u32).map(|i| bits4.contains(&i).into()).collect();
        let p4: BinaryPoly<32> = BinaryPoly::new(coeffs4);
        let r4 = eval_f2_poly_d32_at(&p4, &BinaryFieldGF192::one());
        assert!(r4.is_zero(), "eval at 1 of 4-bit poly should be 0; got {r4}");
    }

    /// `eval(p1 + p2, α) = eval(p1, α) + eval(p2, α)`.
    #[test]
    fn eval_is_linear_over_f2() {
        let alpha = gf(0xDEAD_BEEF_CAFE_F00D, 0xA5A5, 0xBEEF);
        // Two 32-bit-or-less patterns; XOR them in `F_2[X]<32>`.
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

    /// `eval(X · p, α) = α · eval(p, α)`. Multiplying by X shifts every
    /// bit up by one position; we restrict to a polynomial whose top
    /// bit is zero so the shift stays within `<32`.
    #[test]
    fn eval_is_compatible_with_multiplication_by_x() {
        let alpha = gf(0x9E37_79B1, 0xDEAD_BEEF, 0xCAFE_F00D);
        let p_bits: u32 = 0x1234_5678 & 0x7FFF_FFFF; // top bit clear
        let p: BinaryPoly<32> = BinaryPoly::from(p_bits);
        let xp: BinaryPoly<32> = BinaryPoly::from(p_bits << 1);
        let ep = eval_f2_poly_d32_at(&p, &alpha);
        let exp = eval_f2_poly_d32_at(&xp, &alpha);
        assert_eq!(exp, alpha * ep);
    }

    #[test]
    fn implements_field_and_prime_field_traits() {
        // Compile-time check: the synthetic `Field` and `PrimeField`
        // impls on `BinaryFieldGF192` actually satisfy the trait
        // hierarchy. (If a required bound regresses — e.g. someone
        // removes a `CheckedAdd` impl — this assertion fails to
        // compile, which is the point.)
        fn assert_field<F: crypto_primitives::Field>() {}
        fn assert_prime_field<F: crypto_primitives::PrimeField>() {}
        assert_field::<BinaryFieldGF192>();
        assert_prime_field::<BinaryFieldGF192>();
    }

    #[test]
    fn cfg_keyed_constructors_match_const_constructors() {
        use crypto_primitives::PrimeField;
        let cfg = ();
        assert_eq!(
            <BinaryFieldGF192 as PrimeField>::zero_with_cfg(&cfg),
            BinaryFieldGF192::zero(),
        );
        assert_eq!(
            <BinaryFieldGF192 as PrimeField>::one_with_cfg(&cfg),
            BinaryFieldGF192::one(),
        );
        let words = [0xDEAD_BEEFu64, 0xCAFEu64, 0x12345u64];
        let v: BinaryFieldGF192 = <BinaryFieldGF192 as PrimeField>::new_with_cfg(
            Uint::<3>::from_words(words),
            &cfg,
        );
        assert_eq!(*v.words(), words);
    }

    #[test]
    fn division_by_self_yields_one() {
        let a = gf(0xDEAD_BEEF, 0xCAFE_F00D, 0x12345);
        let b = a / a;
        assert!(b.is_one(), "a / a should be 1; got {b}");
    }

    #[test]
    #[should_panic(expected = "GF(2^192) has no prime modulus")]
    fn modulus_minus_one_div_two_panics() {
        use crypto_primitives::PrimeField;
        // This codepath would be reached by prime-field square-root /
        // Legendre-symbol logic — meaningless for a characteristic-2
        // field. The panic documents that misuse.
        let a = BinaryFieldGF192::one();
        let _ = a.modulus_minus_one_div_two();
    }

    /// Lifting then Horner-evaluating must agree with the direct
    /// `eval_f2_poly_d32_at` shortcut — both compute the same ring
    /// homomorphism `F_2[X] → GF(2^192)` (substitute X = α, with the
    /// F_2 → GF(2^192) coefficient embedding).
    #[test]
    fn lift_then_eval_equals_direct_eval() {
        use crate::EvaluatablePolynomial;

        let alpha = gf(0xDEAD_BEEF_CAFE_F00D, 0xA5A5, 0xBEEF);
        for bits in [0u32, 1, 0xA5A5_A5A5, 0xDEAD_BEEF, u32::MAX] {
            let p: BinaryPoly<32> = BinaryPoly::from(bits);
            let direct = eval_f2_poly_d32_at(&p, &alpha);
            let lifted: GF192Poly<32> = lift_f2_poly_to_gf192(&p);
            let horner = lifted.evaluate_at_point(&alpha).unwrap();
            assert_eq!(
                direct, horner,
                "lift-then-eval should equal direct eval for bits = {bits:#x}",
            );
        }
    }

    /// The lift is `F_2`-linear: `lift(p1 + p2) = lift(p1) + lift(p2)`.
    /// This is the algebraic statement that the F_2 → GF(2^192)
    /// coefficient embedding is a ring homomorphism on the coefficient
    /// ring; both `p1 + p2` (in `F_2[X]<32>`, XOR-based) and
    /// `lift(p1) + lift(p2)` (in `GF(2^192)[X]<32>`, coefficient-wise
    /// `GF(2^192)` addition) must produce the same polynomial.
    #[test]
    fn lift_is_linear_over_f2() {
        let p1: BinaryPoly<32> = BinaryPoly::from(0xA5A5_A5A5u32);
        let p2: BinaryPoly<32> = BinaryPoly::from(0xDEAD_BEEFu32);
        let p_sum: BinaryPoly<32> = BinaryPoly::from(0xA5A5_A5A5u32 ^ 0xDEAD_BEEFu32);

        let l1: GF192Poly<32> = lift_f2_poly_to_gf192(&p1);
        let l2: GF192Poly<32> = lift_f2_poly_to_gf192(&p2);
        let l_sum: GF192Poly<32> = lift_f2_poly_to_gf192(&p_sum);

        // GF(2^192)[X] addition is char-2 ⇒ also XOR-style, so the
        // `+` here ends up matching `p1 ^ p2` at the bit level.
        let sum_of_lifts = l1 + l2;
        assert_eq!(sum_of_lifts, l_sum);
    }

    /// `lift_f2_u64_poly_to_gf192` agrees with `lift_f2_poly_to_gf192`
    /// for matching bit patterns — they're two routes to the same
    /// lift, dispatched on the input representation.
    #[test]
    fn lift_u64_and_lift_general_agree() {
        let bits: u32 = 0xCAFE_F00D;
        let p_general: BinaryPoly<32> = BinaryPoly::from(bits);
        let p_u64: BinaryU64Poly<32> = BinaryU64Poly::<32>::from(bits);
        let l_general: GF192Poly<32> = lift_f2_poly_to_gf192(&p_general);
        let l_u64: GF192Poly<32> = lift_f2_u64_poly_to_gf192(&p_u64);
        assert_eq!(l_general, l_u64);
    }

    /// Sanity: the lifted polynomial's coefficients are all 0 or 1
    /// in GF(2^192), never anything else.
    #[test]
    fn lift_coefficients_are_zero_or_one() {
        let p: BinaryPoly<32> = BinaryPoly::from(0xDEAD_BEEFu32);
        let lifted: GF192Poly<32> = lift_f2_poly_to_gf192(&p);
        for c in lifted.coeffs.iter() {
            assert!(
                c.is_zero() || c.is_one(),
                "lifted coefficient must be 0 or 1 in GF(2^192); got {c}",
            );
        }
    }

    #[test]
    fn wide_eval_matches_d32_eval_for_low_degree() {
        let alpha = gf(0xA5A5, 0x5A5A, 0xBEEF);
        let bits: u32 = 0xDEAD_BEEF;
        let p: BinaryPoly<32> = BinaryPoly::from(bits);
        let pw: BinaryF2Poly<1> = BinaryF2Poly::from_words([bits as u64]);
        let e_d32 = eval_f2_poly_d32_at(&p, &alpha);
        let e_wide = eval_f2_wide_poly_at(&pw, &alpha);
        assert_eq!(e_d32, e_wide);
    }
}
