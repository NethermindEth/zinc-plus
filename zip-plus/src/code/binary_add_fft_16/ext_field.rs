//! Verifier-side field `F = F_p[X] / f̃` (= F_{p^16} when f̃ is
//! irreducible mod p) and the 4-bit polynomial challenge type
//! `Bit4Poly16` used as the row-batching `Zt::Chal` in the
//! D=16 polynomial-valued Zip+ variant.
//!
//! Mirrors `binary_add_fft_8/ext_field.rs` but for D = 16.

use core::{
    cmp::Ordering,
    fmt::{self, Debug, Display, Formatter},
    hash::{Hash, Hasher},
    iter::{Product, Sum},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign},
    str::FromStr,
};

use crypto_primitives::{
    Field, FromWithConfig, IntRing, IntSemiring, PrimeField, Ring, Semiring,
    crypto_bigint_int::Int, crypto_bigint_uint::Uint,
};
use num_traits::{
    CheckedAdd, CheckedMul, CheckedNeg, CheckedRem, CheckedSub, ConstOne, ConstZero, Inv, One, Pow,
    Zero,
};
use zinc_poly::univariate::{binary::BinaryPoly, dense::DensePolynomial};
use zinc_transcript::traits::{ConstTranscribable, GenTranscribable};
use zinc_utils::{from_ref::FromRef, mul_by_scalar::MulByScalar, named::Named};

use super::basis::{D_16, F_TILDE_LOWER_DEGREES_16};

// ===========================================================================
// Mod-p primitive helpers (P ≤ 2^32 keeps products in u64).
// ===========================================================================

#[inline]
const fn add_mod_p(a: u64, b: u64, p: u64) -> u64 {
    let s = a + b;
    if s >= p { s - p } else { s }
}

#[inline]
const fn sub_mod_p(a: u64, b: u64, p: u64) -> u64 {
    if a >= b { a - b } else { p - (b - a) }
}

#[inline]
const fn neg_mod_p(a: u64, p: u64) -> u64 {
    if a == 0 { 0 } else { p - a }
}

#[inline]
const fn mul_mod_p(a: u64, b: u64, p: u64) -> u64 {
    if p < (1u64 << 32) {
        (a.wrapping_mul(b)) % p
    } else {
        let prod = (a as u128) * (b as u128);
        (prod % (p as u128)) as u64
    }
}

/// Const-generic mul_mod_p with Barrett reduction.
#[inline(always)]
const fn mul_mod_p_cg<const P: u64>(a: u64, b: u64) -> u64 {
    if P < (1u64 << 32) {
        let x = a.wrapping_mul(b);
        let m: u64 = ((1u128 << 64) / (P as u128)) as u64;
        let q = ((x as u128).wrapping_mul(m as u128) >> 64) as u64;
        let mut r = x.wrapping_sub(q.wrapping_mul(P));
        if r >= P {
            r -= P;
        }
        r
    } else {
        ((a as u128) * (b as u128) % (P as u128)) as u64
    }
}

#[inline(always)]
const fn add_mod_p_cg<const P: u64>(a: u64, b: u64) -> u64 {
    let s = a + b;
    if s >= P { s - P } else { s }
}

/// Reduce a length-(2D-1) buffer mod f̃ into a length-D array.
#[inline]
fn reduce_mod_ftilde_p<const P: u64>(buf: &[u64; 2 * D_16 - 1]) -> [u64; D_16] {
    let mut local = *buf;
    for j in (D_16..(2 * D_16 - 1)).rev() {
        let c = local[j];
        if c == 0 {
            continue;
        }
        local[j] = 0;
        let base = j - D_16;
        for &shift in &F_TILDE_LOWER_DEGREES_16 {
            local[base + shift] = add_mod_p_cg::<P>(local[base + shift], c);
        }
    }
    let mut out = [0u64; D_16];
    out.copy_from_slice(&local[..D_16]);
    out
}

#[inline]
fn red(coeff: u64, p: u64) -> u64 {
    if coeff < p { coeff } else { coeff % p }
}

/// Modular inverse of `a` (mod `p`, p prime).
pub fn inv_mod_p(a: u64, p: u64) -> Option<u64> {
    if a == 0 {
        return None;
    }
    let mut result = 1u64;
    let mut base = a;
    let mut exp = p - 2;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mul_mod_p(result, base, p);
        }
        exp >>= 1;
        if exp > 0 {
            base = mul_mod_p(base, base, p);
        }
    }
    Some(result)
}

// ===========================================================================
// Gf2_16ExtInner — newtype wrapper for F::Inner.
// ===========================================================================

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Gf2_16ExtInner(pub [u64; D_16]);

impl GenTranscribable for Gf2_16ExtInner {
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        let mut out = [0u64; D_16];
        for i in 0..D_16 {
            let mut b8 = [0u8; 8];
            b8.copy_from_slice(&bytes[i * 8..(i + 1) * 8]);
            out[i] = u64::from_le_bytes(b8);
        }
        Self(out)
    }

    fn write_transcription_bytes_exact(&self, buf: &mut [u8]) {
        for (i, limb) in self.0.iter().enumerate() {
            buf[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
        }
    }
}

impl ConstTranscribable for Gf2_16ExtInner {
    const NUM_BYTES: usize = D_16 * 8;
    const NUM_BITS: usize = D_16 * 64;
}

// ===========================================================================
// Gf2_16Ext<P> — F_p[X] / f̃.
// ===========================================================================

#[derive(Clone, Copy)]
pub struct Gf2_16Ext<const P: u64> {
    pub inner: Gf2_16ExtInner,
}

impl<const P: u64> Gf2_16Ext<P> {
    pub const ZERO_VAL: Self = Self { inner: Gf2_16ExtInner([0; D_16]) };

    pub const ONE_VAL: Self = {
        let mut c = [0u64; D_16];
        c[0] = 1;
        Self { inner: Gf2_16ExtInner(c) }
    };

    #[inline]
    pub fn new_array(coeffs: [u64; D_16]) -> Self {
        let mut out = [0u64; D_16];
        let mut i = 0;
        while i < D_16 {
            out[i] = red(coeffs[i], P);
            i += 1;
        }
        Self { inner: Gf2_16ExtInner(out) }
    }

    #[inline]
    pub fn new_unchecked_array(coeffs: [u64; D_16]) -> Self {
        Self { inner: Gf2_16ExtInner(coeffs) }
    }

    #[inline]
    pub fn coeffs(&self) -> &[u64; D_16] {
        &self.inner.0
    }

    /// Schoolbook poly mult of two reduced (length-D) inputs, then mod-f̃.
    fn mul_inner(a: &[u64; D_16], b: &[u64; D_16]) -> Self {
        let mut prod = [0u64; 2 * D_16 - 1];
        for i in 0..D_16 {
            let ai = a[i];
            if ai == 0 {
                continue;
            }
            for j in 0..D_16 {
                let bj = b[j];
                if bj == 0 {
                    continue;
                }
                prod[i + j] = add_mod_p_cg::<P>(prod[i + j], mul_mod_p_cg::<P>(ai, bj));
            }
        }
        let out = reduce_mod_ftilde_p::<P>(&prod);
        Self { inner: Gf2_16ExtInner(out) }
    }

    /// Polynomial inversion via extended Euclidean in F_p[X].
    pub fn try_inv(&self) -> Option<Self> {
        let mut f_tilde = vec![0u64; D_16 + 1];
        f_tilde[D_16] = 1;
        for &k in &F_TILDE_LOWER_DEGREES_16 {
            f_tilde[k] = 1;
        }

        let mut r0 = f_tilde;
        let mut r1: Vec<u64> = self.coeffs().to_vec();
        while r1.len() > 1 && *r1.last().unwrap() == 0 {
            r1.pop();
        }
        if r1.last().copied() == Some(0) {
            return None;
        }
        let mut s0: Vec<u64> = vec![0];
        let mut s1: Vec<u64> = vec![1];

        while !(r1.len() == 1 && r1[0] == 0) {
            let (q, rem) = poly_divmod(&r0, &r1, P);
            let qs1 = poly_mul(&q, &s1, P);
            let new_s = poly_sub(&s0, &qs1, P);
            r0 = r1;
            r1 = rem;
            s0 = s1;
            s1 = new_s;
        }
        if r0.len() != 1 || r0[0] == 0 {
            return None;
        }
        let inv_g = inv_mod_p(r0[0], P)?;
        let mut out = [0u64; D_16];
        for (i, c) in s0.iter().enumerate().take(D_16) {
            out[i] = mul_mod_p(*c, inv_g, P);
        }
        Some(Self { inner: Gf2_16ExtInner(out) })
    }

    /// Rabin irreducibility test for f̃ over F_P.
    /// For D = 16, the only Rabin-relevant maximal proper divisor of 16
    /// (which is 16/q for prime q | 16) is 16/2 = 8. So we check
    /// (a) X^{P^16} ≡ X mod f̃, and (b) gcd(f̃, X^{P^8} - X) = 1.
    pub fn f_tilde_is_irreducible_mod_p() -> bool {
        let x = {
            let mut c = [0u64; D_16];
            c[1] = 1;
            Self::new_unchecked_array(c)
        };
        let xp16 = x_to_p_power_k(&x, 16);
        if xp16.coeffs() != x.coeffs() {
            return false;
        }
        let xp8 = x_to_p_power_k(&x, 8);
        let mut diff = *xp8.coeffs();
        diff[1] = sub_mod_p(diff[1], 1, P);
        let mut diff_vec: Vec<u64> = diff.to_vec();
        while diff_vec.len() > 1 && *diff_vec.last().unwrap() == 0 {
            diff_vec.pop();
        }
        if diff_vec == [0] {
            return false;
        }
        let mut f_tilde = vec![0u64; D_16 + 1];
        f_tilde[D_16] = 1;
        for &k in &F_TILDE_LOWER_DEGREES_16 {
            f_tilde[k] = 1;
        }
        let g = poly_gcd(&f_tilde, &diff_vec, P);
        g.len() == 1 && g[0] != 0
    }
}

fn x_to_p_power_k<const P: u64>(x: &Gf2_16Ext<P>, k: u32) -> Gf2_16Ext<P> {
    let mut acc = *x;
    for _ in 0..k {
        let mut result = Gf2_16Ext::<P>::ONE_VAL;
        let mut base = acc;
        let mut exp = P;
        while exp > 0 {
            if exp & 1 == 1 {
                result = Gf2_16Ext::<P>::mul_inner(result.coeffs(), base.coeffs());
            }
            exp >>= 1;
            if exp > 0 {
                base = Gf2_16Ext::<P>::mul_inner(base.coeffs(), base.coeffs());
            }
        }
        acc = result;
    }
    acc
}

// ---- F_p[X] helpers -------------------------------------------------------

fn poly_sub(a: &[u64], b: &[u64], p: u64) -> Vec<u64> {
    let n = a.len().max(b.len());
    let mut out = vec![0u64; n];
    for i in 0..n {
        let ai = if i < a.len() { a[i] } else { 0 };
        let bi = if i < b.len() { b[i] } else { 0 };
        out[i] = sub_mod_p(ai, bi, p);
    }
    while out.len() > 1 && *out.last().unwrap() == 0 {
        out.pop();
    }
    out
}

fn poly_mul(a: &[u64], b: &[u64], p: u64) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return vec![0];
    }
    let mut out = vec![0u64; a.len() + b.len() - 1];
    for i in 0..a.len() {
        if a[i] == 0 {
            continue;
        }
        for j in 0..b.len() {
            if b[j] == 0 {
                continue;
            }
            out[i + j] = add_mod_p(out[i + j], mul_mod_p(a[i], b[j], p), p);
        }
    }
    out
}

fn poly_divmod(num: &[u64], denom: &[u64], p: u64) -> (Vec<u64>, Vec<u64>) {
    let mut denom = denom.to_vec();
    while denom.len() > 1 && *denom.last().unwrap() == 0 {
        denom.pop();
    }
    let mut rem = num.to_vec();
    while rem.len() > 1 && *rem.last().unwrap() == 0 {
        rem.pop();
    }
    if rem.len() < denom.len() {
        return (vec![0], rem);
    }
    let lead_denom_inv = inv_mod_p(*denom.last().unwrap(), p)
        .expect("denom leading coeff non-zero & invertible");
    let mut q = vec![0u64; rem.len() - denom.len() + 1];
    while rem.len() >= denom.len() && !(rem.len() == 1 && rem[0] == 0) {
        let lead = *rem.last().unwrap();
        if lead == 0 {
            rem.pop();
            continue;
        }
        let coeff = mul_mod_p(lead, lead_denom_inv, p);
        let pos = rem.len() - denom.len();
        q[pos] = coeff;
        for i in 0..denom.len() {
            let sub_val = mul_mod_p(coeff, denom[i], p);
            rem[pos + i] = sub_mod_p(rem[pos + i], sub_val, p);
        }
        while rem.len() > 1 && *rem.last().unwrap() == 0 {
            rem.pop();
        }
    }
    (q, rem)
}

fn poly_gcd(a: &[u64], b: &[u64], p: u64) -> Vec<u64> {
    let mut r0 = a.to_vec();
    let mut r1 = b.to_vec();
    while !(r1.len() == 1 && r1[0] == 0) {
        let (_q, rem) = poly_divmod(&r0, &r1, p);
        r0 = r1;
        r1 = rem;
    }
    if r0.is_empty() || (r0.len() == 1 && r0[0] == 0) {
        return vec![0];
    }
    let inv_lead = inv_mod_p(*r0.last().unwrap(), p).unwrap();
    for c in r0.iter_mut() {
        *c = mul_mod_p(*c, inv_lead, p);
    }
    r0
}

// ===========================================================================
// Default prime P for Gf2_16Ext.
//
// Chosen by the `find_irreducible_prime_scan` test below: f̃_16 =
// X^16 + X^5 + X^3 + X^2 + 1 is irreducible mod 2_147_482_921 (the
// largest prime strictly below 2^31 for which Rabin's test for D=16
// passes — none of the standard NTT-friendly primes 998_244_353 /
// 2_013_265_921 / 1_811_939_329 / 1_004_535_809 satisfy this, and the
// Mersenne prime 2^31 - 1 also fails). This near-maximal prime gives
// strong security for Fiat-Shamir random-element sampling while
// keeping products in u64.
// ===========================================================================

/// Default prime P for which f̃_16 is irreducible (so `Gf2_16Ext<P_16_DEFAULT>`
/// is the genuine extension field F_{p^16}). Selected by Rabin's test.
pub const P_16_DEFAULT: u64 = 2_147_482_921;

// ===========================================================================
// Trait impls for Gf2_16Ext.
// ===========================================================================

impl<const P: u64> PartialEq for Gf2_16Ext<P> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}
impl<const P: u64> Eq for Gf2_16Ext<P> {}
impl<const P: u64> Hash for Gf2_16Ext<P> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}
impl<const P: u64> Debug for Gf2_16Ext<P> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Gf2_16Ext<{}>({:?})", P, &self.inner.0)
    }
}
impl<const P: u64> Display for Gf2_16Ext<P> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Gf2_16Ext<{}>(...)", P)
    }
}
impl<const P: u64> Default for Gf2_16Ext<P> {
    fn default() -> Self {
        Self::ZERO_VAL
    }
}
impl<const P: u64> PartialOrd for Gf2_16Ext<P> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.inner.0.cmp(&other.inner.0))
    }
}
impl<const P: u64> Ord for Gf2_16Ext<P> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.inner.0.cmp(&other.inner.0)
    }
}

impl<const P: u64> ConstZero for Gf2_16Ext<P> {
    const ZERO: Self = Self::ZERO_VAL;
}
impl<const P: u64> Zero for Gf2_16Ext<P> {
    fn zero() -> Self {
        Self::ZERO_VAL
    }
    fn is_zero(&self) -> bool {
        self.inner.0.iter().all(|&c| c == 0)
    }
}
impl<const P: u64> ConstOne for Gf2_16Ext<P> {
    const ONE: Self = Self::ONE_VAL;
}
impl<const P: u64> One for Gf2_16Ext<P> {
    fn one() -> Self {
        Self::ONE_VAL
    }
}

impl<const P: u64> Neg for Gf2_16Ext<P> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        let mut out = [0u64; D_16];
        for i in 0..D_16 {
            out[i] = neg_mod_p(self.inner.0[i], P);
        }
        Self { inner: Gf2_16ExtInner(out) }
    }
}

impl<const P: u64> Add for Gf2_16Ext<P> {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self::Output {
        self += &rhs;
        self
    }
}
impl<const P: u64> Add<&Self> for Gf2_16Ext<P> {
    type Output = Self;
    fn add(mut self, rhs: &Self) -> Self::Output {
        self += rhs;
        self
    }
}
impl<const P: u64> AddAssign for Gf2_16Ext<P> {
    fn add_assign(&mut self, rhs: Self) {
        *self += &rhs;
    }
}
impl<const P: u64> AddAssign<&Self> for Gf2_16Ext<P> {
    fn add_assign(&mut self, rhs: &Self) {
        for i in 0..D_16 {
            self.inner.0[i] = add_mod_p_cg::<P>(self.inner.0[i], rhs.inner.0[i]);
        }
    }
}

impl<const P: u64> Sub for Gf2_16Ext<P> {
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= &rhs;
        self
    }
}
impl<const P: u64> Sub<&Self> for Gf2_16Ext<P> {
    type Output = Self;
    fn sub(mut self, rhs: &Self) -> Self::Output {
        self -= rhs;
        self
    }
}
impl<const P: u64> SubAssign for Gf2_16Ext<P> {
    fn sub_assign(&mut self, rhs: Self) {
        *self -= &rhs;
    }
}
impl<const P: u64> SubAssign<&Self> for Gf2_16Ext<P> {
    fn sub_assign(&mut self, rhs: &Self) {
        for i in 0..D_16 {
            self.inner.0[i] = sub_mod_p(self.inner.0[i], rhs.inner.0[i], P);
        }
    }
}

impl<const P: u64> Mul for Gf2_16Ext<P> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self::mul_inner(self.coeffs(), rhs.coeffs())
    }
}
impl<const P: u64> Mul<&Self> for Gf2_16Ext<P> {
    type Output = Self;
    fn mul(self, rhs: &Self) -> Self::Output {
        Self::mul_inner(self.coeffs(), rhs.coeffs())
    }
}
impl<const P: u64> MulAssign for Gf2_16Ext<P> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = Self::mul_inner(self.coeffs(), rhs.coeffs());
    }
}
impl<const P: u64> MulAssign<&Self> for Gf2_16Ext<P> {
    fn mul_assign(&mut self, rhs: &Self) {
        *self = Self::mul_inner(self.coeffs(), rhs.coeffs());
    }
}

impl<const P: u64> Div for Gf2_16Ext<P> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        let inv = rhs.try_inv().expect("Gf2_16Ext::div by non-invertible element");
        self * inv
    }
}
impl<const P: u64> Div<&Self> for Gf2_16Ext<P> {
    type Output = Self;
    fn div(self, rhs: &Self) -> Self::Output {
        let inv = rhs.try_inv().expect("Gf2_16Ext::div by non-invertible element");
        self * inv
    }
}
impl<const P: u64> DivAssign for Gf2_16Ext<P> {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}
impl<const P: u64> DivAssign<&Self> for Gf2_16Ext<P> {
    fn div_assign(&mut self, rhs: &Self) {
        *self = *self / rhs;
    }
}

impl<const P: u64> Pow<u32> for Gf2_16Ext<P> {
    type Output = Self;
    fn pow(self, mut exp: u32) -> Self::Output {
        let mut result = Self::ONE_VAL;
        let mut base = self;
        while exp > 0 {
            if exp & 1 == 1 {
                result = Self::mul_inner(result.coeffs(), base.coeffs());
            }
            exp >>= 1;
            if exp > 0 {
                base = Self::mul_inner(base.coeffs(), base.coeffs());
            }
        }
        result
    }
}

impl<const P: u64> Inv for Gf2_16Ext<P> {
    type Output = Option<Self>;
    fn inv(self) -> Self::Output {
        self.try_inv()
    }
}

impl<const P: u64> CheckedAdd for Gf2_16Ext<P> {
    fn checked_add(&self, rhs: &Self) -> Option<Self> {
        Some(*self + rhs)
    }
}
impl<const P: u64> CheckedSub for Gf2_16Ext<P> {
    fn checked_sub(&self, rhs: &Self) -> Option<Self> {
        Some(*self - rhs)
    }
}
impl<const P: u64> CheckedMul for Gf2_16Ext<P> {
    fn checked_mul(&self, rhs: &Self) -> Option<Self> {
        Some(*self * rhs)
    }
}
impl<const P: u64> CheckedNeg for Gf2_16Ext<P> {
    fn checked_neg(&self) -> Option<Self> {
        Some(-*self)
    }
}

impl<const P: u64> Sum for Gf2_16Ext<P> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO_VAL, |acc, x| acc + x)
    }
}
impl<'a, const P: u64> Sum<&'a Self> for Gf2_16Ext<P> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO_VAL, |acc, x| acc + x)
    }
}
impl<const P: u64> Product for Gf2_16Ext<P> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE_VAL, |acc, x| acc * x)
    }
}
impl<'a, const P: u64> Product<&'a Self> for Gf2_16Ext<P> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ONE_VAL, |acc, x| acc * x)
    }
}

impl<const P: u64> Semiring for Gf2_16Ext<P> {}
impl<const P: u64> Ring for Gf2_16Ext<P> {}

impl<const P: u64> Field for Gf2_16Ext<P> {
    type Inner = Gf2_16ExtInner;
    // Width-matched to Inner: D*8 = 128 bytes. The protocol's
    // `write_field_element_no_length` reuses a single buffer sized at
    // `Inner::NUM_BYTES` for BOTH modulus and inner writes, so they
    // must share a byte width.
    type Modulus = Uint<16>;

    fn inner(&self) -> &Self::Inner {
        &self.inner
    }
    fn inner_mut(&mut self) -> &mut Self::Inner {
        &mut self.inner
    }
    fn into_inner(self) -> Self::Inner {
        self.inner
    }
}

impl<const P: u64> PrimeField for Gf2_16Ext<P> {
    type Config = ();
    fn cfg(&self) -> &Self::Config {
        &()
    }
    fn is_zero(value: &Self) -> bool {
        Zero::is_zero(value)
    }
    fn modulus(&self) -> Self::Modulus {
        Uint::<16>::from(P)
    }
    fn modulus_minus_one_div_two(&self) -> Self::Inner {
        Gf2_16ExtInner([0u64; D_16])
    }
    fn make_cfg(modulus: &Self::Modulus) -> Result<Self::Config, crypto_primitives::FieldError> {
        if *modulus == Uint::<16>::from(P) {
            Ok(())
        } else {
            Err(crypto_primitives::FieldError::InvalidModulus)
        }
    }
    fn new_with_cfg(inner: Self::Inner, _cfg: &Self::Config) -> Self {
        Self::new_array(inner.0)
    }
    fn new_unchecked_with_cfg(inner: Self::Inner, _cfg: &Self::Config) -> Self {
        Self { inner }
    }
    fn zero_with_cfg(_cfg: &Self::Config) -> Self {
        Self::ZERO_VAL
    }
    fn one_with_cfg(_cfg: &Self::Config) -> Self {
        Self::ONE_VAL
    }
}

impl<const P: u64> FromRef<Gf2_16Ext<P>> for Gf2_16Ext<P> {
    fn from_ref(value: &Self) -> Self {
        *value
    }
}

impl<const P: u64> FromWithConfig<&Gf2_16Ext<P>> for Gf2_16Ext<P> {
    fn from_with_cfg(value: &Self, _cfg: &Self::Config) -> Self {
        *value
    }
}

impl<const P: u64, const N: usize> FromWithConfig<&DensePolynomial<Int<N>, 16>>
    for Gf2_16Ext<P>
{
    fn from_with_cfg(
        value: &DensePolynomial<Int<N>, 16>,
        _cfg: &Self::Config,
    ) -> Self {
        let mut out = [0u64; D_16];
        for (i, c) in value.coeffs.iter().enumerate() {
            out[i] = int_to_u64_mod_p::<N>(c, P);
        }
        Self { inner: Gf2_16ExtInner(out) }
    }
}

impl<const P: u64> FromWithConfig<&BinaryPoly<16>> for Gf2_16Ext<P> {
    fn from_with_cfg(value: &BinaryPoly<16>, _cfg: &Self::Config) -> Self {
        let mut out = [0u64; D_16];
        for (i, b) in value.iter().enumerate().take(D_16) {
            if b.into_inner() {
                out[i] = 1;
            }
        }
        Self { inner: Gf2_16ExtInner(out) }
    }
}

impl<const P: u64> FromWithConfig<&DensePolynomial<i64, 16>> for Gf2_16Ext<P> {
    fn from_with_cfg(value: &DensePolynomial<i64, 16>, _cfg: &Self::Config) -> Self {
        let mut out = [0u64; D_16];
        let pp = P as i128;
        for (i, c) in value.coeffs.iter().enumerate() {
            out[i] = (*c as i128).rem_euclid(pp) as u64;
        }
        Self { inner: Gf2_16ExtInner(out) }
    }
}

impl<const P: u64> FromWithConfig<&DensePolynomial<i128, 16>> for Gf2_16Ext<P> {
    fn from_with_cfg(value: &DensePolynomial<i128, 16>, _cfg: &Self::Config) -> Self {
        let mut out = [0u64; D_16];
        let pp = P as i128;
        for (i, c) in value.coeffs.iter().enumerate() {
            out[i] = c.rem_euclid(pp) as u64;
        }
        Self { inner: Gf2_16ExtInner(out) }
    }
}

fn int_to_u64_mod_p<const N: usize>(value: &Int<N>, p: u64) -> u64 {
    let abs = match value.checked_abs() {
        Some(a) => a,
        None => {
            return int_min_to_u64_mod_p::<N>(p);
        }
    };
    let mut acc = 0u128;
    let pp = p as u128;
    let limbs = abs.inner().to_words();
    for limb in limbs.iter().rev() {
        acc = (acc << 64) % pp;
        acc = (acc + (*limb as u128)) % pp;
    }
    let mag = acc as u64;
    if value.is_negative() {
        if mag == 0 { 0 } else { p - mag }
    } else {
        mag
    }
}

fn int_min_to_u64_mod_p<const N: usize>(p: u64) -> u64 {
    let mut acc: u128 = 1;
    let total_bits = 64 * N - 1;
    let pp = p as u128;
    for _ in 0..total_bits {
        acc = (acc << 1) % pp;
    }
    let mag = acc as u64;
    if mag == 0 { 0 } else { p - mag }
}

impl<const P: u64> MulByScalar<&Gf2_16Ext<P>> for Gf2_16Ext<P> {
    fn mul_by_scalar<const CHECK: bool>(&self, rhs: &Self) -> Option<Self> {
        Some(*self * rhs)
    }
}

macro_rules! impl_from_unsigned_for_ext {
    ($($t:ty),*) => {
        $(
            impl<const P: u64> From<$t> for Gf2_16Ext<P> {
                fn from(value: $t) -> Self {
                    let mut c = [0u64; D_16];
                    c[0] = (value as u128 % P as u128) as u64;
                    Self { inner: Gf2_16ExtInner(c) }
                }
            }
        )*
    };
}
macro_rules! impl_from_signed_for_ext {
    ($($t:ty),*) => {
        $(
            impl<const P: u64> From<$t> for Gf2_16Ext<P> {
                fn from(value: $t) -> Self {
                    let mut c = [0u64; D_16];
                    let pp = P as i128;
                    let v = value as i128;
                    let r = ((v % pp) + pp) % pp;
                    c[0] = r as u64;
                    Self { inner: Gf2_16ExtInner(c) }
                }
            }
        )*
    };
}
impl_from_unsigned_for_ext!(u8, u16, u32, u64, u128);
impl_from_signed_for_ext!(i8, i16, i32, i64, i128);

// ===========================================================================
// Bit4Poly16: row-batching `Chal`.
//
// 16 nibbles × 4 bits = 64 bits, so we use `i64` as the backing
// integer. Standard ring arithmetic is inherited from i64. The
// polynomial interpretation only matters where the nibble view is
// queried (MulByScalar impls, FromWithConfig for Gf2_16Ext).
// ===========================================================================

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Debug)]
#[repr(transparent)]
pub struct Bit4Poly16(pub i64);

impl Bit4Poly16 {
    /// Read the 16 nibbles (low nibble first) as polynomial coefficients.
    #[inline]
    pub fn nibbles(&self) -> [u8; D_16] {
        let mut out = [0u8; D_16];
        let bytes = self.0.to_le_bytes();
        for i in 0..D_16 / 2 {
            out[2 * i] = bytes[i] & 0x0F;
            out[2 * i + 1] = (bytes[i] >> 4) & 0x0F;
        }
        out
    }
}

impl Display for Bit4Poly16 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Bit4Poly16({:#018x})", self.0)
    }
}

impl Named for Bit4Poly16 {
    fn type_name() -> String {
        "Bit4Poly16".to_string()
    }
}

// --- Numeric trait suite, delegated to i64 ---------------------------------

macro_rules! delegate_unary {
    ($trait:ident, $method:ident) => {
        impl $trait for Bit4Poly16 {
            type Output = Self;
            fn $method(self) -> Self::Output {
                Self(self.0.$method())
            }
        }
    };
}
macro_rules! delegate_binary {
    ($trait:ident, $method:ident) => {
        impl $trait for Bit4Poly16 {
            type Output = Self;
            #[allow(clippy::arithmetic_side_effects)]
            fn $method(self, rhs: Self) -> Self::Output {
                Self(self.0.$method(rhs.0))
            }
        }
        impl $trait<&Self> for Bit4Poly16 {
            type Output = Self;
            #[allow(clippy::arithmetic_side_effects)]
            fn $method(self, rhs: &Self) -> Self::Output {
                Self(self.0.$method(rhs.0))
            }
        }
    };
}
macro_rules! delegate_binary_assign {
    ($trait:ident, $method:ident) => {
        impl $trait for Bit4Poly16 {
            #[allow(clippy::arithmetic_side_effects)]
            fn $method(&mut self, rhs: Self) {
                self.0.$method(rhs.0);
            }
        }
        impl $trait<&Self> for Bit4Poly16 {
            #[allow(clippy::arithmetic_side_effects)]
            fn $method(&mut self, rhs: &Self) {
                self.0.$method(rhs.0);
            }
        }
    };
}

delegate_unary!(Neg, neg);
delegate_binary!(Add, add);
delegate_binary!(Sub, sub);
delegate_binary!(Mul, mul);
delegate_binary!(Div, div);
delegate_binary!(Rem, rem);
delegate_binary_assign!(AddAssign, add_assign);
delegate_binary_assign!(SubAssign, sub_assign);
delegate_binary_assign!(MulAssign, mul_assign);
delegate_binary_assign!(DivAssign, div_assign);
delegate_binary_assign!(RemAssign, rem_assign);

impl Zero for Bit4Poly16 {
    fn zero() -> Self {
        Self(0)
    }
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}
impl ConstZero for Bit4Poly16 {
    const ZERO: Self = Self(0);
}
impl One for Bit4Poly16 {
    fn one() -> Self {
        Self(1)
    }
}
impl ConstOne for Bit4Poly16 {
    const ONE: Self = Self(1);
}
impl Pow<u32> for Bit4Poly16 {
    type Output = Self;
    #[allow(clippy::arithmetic_side_effects)]
    fn pow(self, exp: u32) -> Self::Output {
        Self(i64::pow(self.0, exp))
    }
}

impl CheckedAdd for Bit4Poly16 {
    fn checked_add(&self, rhs: &Self) -> Option<Self> {
        self.0.checked_add(rhs.0).map(Self)
    }
}
impl CheckedSub for Bit4Poly16 {
    fn checked_sub(&self, rhs: &Self) -> Option<Self> {
        self.0.checked_sub(rhs.0).map(Self)
    }
}
impl CheckedMul for Bit4Poly16 {
    fn checked_mul(&self, rhs: &Self) -> Option<Self> {
        self.0.checked_mul(rhs.0).map(Self)
    }
}
impl CheckedNeg for Bit4Poly16 {
    fn checked_neg(&self) -> Option<Self> {
        self.0.checked_neg().map(Self)
    }
}
impl CheckedRem for Bit4Poly16 {
    fn checked_rem(&self, rhs: &Self) -> Option<Self> {
        self.0.checked_rem(rhs.0).map(Self)
    }
}

impl Sum for Bit4Poly16 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        Self(iter.map(|x| x.0).sum())
    }
}
impl<'a> Sum<&'a Self> for Bit4Poly16 {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        Self(iter.map(|x| x.0).sum())
    }
}
impl Product for Bit4Poly16 {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        Self(iter.map(|x| x.0).product())
    }
}
impl<'a> Product<&'a Self> for Bit4Poly16 {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        Self(iter.map(|x| x.0).product())
    }
}

impl From<bool> for Bit4Poly16 {
    fn from(v: bool) -> Self {
        Self(v as i64)
    }
}
impl From<i8> for Bit4Poly16 {
    fn from(v: i8) -> Self {
        Self(v as i64)
    }
}
impl FromStr for Bit4Poly16 {
    type Err = <i64 as FromStr>::Err;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.parse::<i64>().map(Self)
    }
}

impl Semiring for Bit4Poly16 {}
impl Ring for Bit4Poly16 {}
impl crypto_primitives::ConstSemiring for Bit4Poly16 {
    const MIN: Self = Self(i64::MIN);
    const MAX: Self = Self(i64::MAX);
}
impl IntSemiring for Bit4Poly16 {
    fn is_odd(&self) -> bool {
        self.0 & 1 == 1
    }
    fn is_even(&self) -> bool {
        self.0 & 1 == 0
    }
}
impl IntRing for Bit4Poly16 {
    fn checked_abs(&self) -> Option<Self> {
        self.0.checked_abs().map(Self)
    }
    fn is_positive(&self) -> bool {
        self.0 > 0
    }
    fn is_negative(&self) -> bool {
        self.0 < 0
    }
}

impl GenTranscribable for Bit4Poly16 {
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        let mut b = [0u8; 8];
        b.copy_from_slice(&bytes[..8]);
        Self(i64::from_le_bytes(b))
    }
    fn write_transcription_bytes_exact(&self, buf: &mut [u8]) {
        buf[..8].copy_from_slice(&self.0.to_le_bytes());
    }
}
impl ConstTranscribable for Bit4Poly16 {
    const NUM_BYTES: usize = 8;
    const NUM_BITS: usize = 64;
}

// --- The two polynomial-interpretation impls -------------------------------

impl<const P: u64> FromWithConfig<&Bit4Poly16> for Gf2_16Ext<P> {
    fn from_with_cfg(value: &Bit4Poly16, _cfg: &Self::Config) -> Self {
        let nibbles = value.nibbles();
        let mut out = [0u64; D_16];
        for i in 0..D_16 {
            out[i] = nibbles[i] as u64;
        }
        Self { inner: Gf2_16ExtInner(out) }
    }
}

impl<const M: usize> MulByScalar<&Bit4Poly16> for DensePolynomial<Int<M>, 16> {
    fn mul_by_scalar<const CHECK: bool>(&self, rhs: &Bit4Poly16) -> Option<Self> {
        let nibbles = rhs.nibbles();
        let lhs = &self.coeffs;
        let mut prod = [Int::<M>::ZERO; 2 * D_16 - 1];
        for i in 0..D_16 {
            let ai = lhs[i];
            if ai == Int::<M>::ZERO {
                continue;
            }
            for j in 0..D_16 {
                let bj = nibbles[j] as i64;
                if bj == 0 {
                    continue;
                }
                let scaled = ai.mul_by_scalar::<CHECK>(&bj)?;
                prod[i + j] = prod[i + j] + scaled;
            }
        }
        for j in (D_16..(2 * D_16 - 1)).rev() {
            let c = prod[j];
            if c == Int::<M>::ZERO {
                continue;
            }
            prod[j] = Int::<M>::ZERO;
            let base = j - D_16;
            for &shift in &F_TILDE_LOWER_DEGREES_16 {
                prod[base + shift] = prod[base + shift] + c;
            }
        }
        let mut out = [Int::<M>::ZERO; D_16];
        out.copy_from_slice(&prod[..D_16]);
        Some(DensePolynomial::new(out))
    }
}

impl MulByScalar<&Bit4Poly16> for DensePolynomial<i64, 16> {
    fn mul_by_scalar<const CHECK: bool>(&self, rhs: &Bit4Poly16) -> Option<Self> {
        let nibbles = rhs.nibbles();
        let lhs = &self.coeffs;
        let mut prod = [0i64; 2 * D_16 - 1];
        for i in 0..D_16 {
            let ai = lhs[i];
            if ai == 0 {
                continue;
            }
            for j in 0..D_16 {
                let bj = nibbles[j] as i64;
                if bj == 0 {
                    continue;
                }
                let scaled = if CHECK {
                    ai.checked_mul(bj)?
                } else {
                    ai.wrapping_mul(bj)
                };
                prod[i + j] = if CHECK {
                    prod[i + j].checked_add(scaled)?
                } else {
                    prod[i + j].wrapping_add(scaled)
                };
            }
        }
        for j in (D_16..(2 * D_16 - 1)).rev() {
            let c = prod[j];
            if c == 0 {
                continue;
            }
            prod[j] = 0;
            let base = j - D_16;
            for &shift in &F_TILDE_LOWER_DEGREES_16 {
                prod[base + shift] = if CHECK {
                    prod[base + shift].checked_add(c)?
                } else {
                    prod[base + shift].wrapping_add(c)
                };
            }
        }
        let mut out = [0i64; D_16];
        out.copy_from_slice(&prod[..D_16]);
        Some(DensePolynomial::new(out))
    }
}

impl MulByScalar<&Bit4Poly16> for DensePolynomial<i128, 16> {
    fn mul_by_scalar<const CHECK: bool>(&self, rhs: &Bit4Poly16) -> Option<Self> {
        let nibbles = rhs.nibbles();
        let lhs = &self.coeffs;
        let mut prod = [0i128; 2 * D_16 - 1];
        for i in 0..D_16 {
            let ai = lhs[i];
            if ai == 0 {
                continue;
            }
            for j in 0..D_16 {
                let bj = nibbles[j] as i128;
                if bj == 0 {
                    continue;
                }
                let scaled = if CHECK {
                    ai.checked_mul(bj)?
                } else {
                    ai.wrapping_mul(bj)
                };
                prod[i + j] = if CHECK {
                    prod[i + j].checked_add(scaled)?
                } else {
                    prod[i + j].wrapping_add(scaled)
                };
            }
        }
        for j in (D_16..(2 * D_16 - 1)).rev() {
            let c = prod[j];
            if c == 0 {
                continue;
            }
            prod[j] = 0;
            let base = j - D_16;
            for &shift in &F_TILDE_LOWER_DEGREES_16 {
                prod[base + shift] = if CHECK {
                    prod[base + shift].checked_add(c)?
                } else {
                    prod[base + shift].wrapping_add(c)
                };
            }
        }
        let mut out = [0i128; D_16];
        out.copy_from_slice(&prod[..D_16]);
        Some(DensePolynomial::new(out))
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn is_prime(n: u64) -> bool {
        if n < 2 {
            return false;
        }
        if n % 2 == 0 {
            return n == 2;
        }
        let mut i = 3u64;
        while i * i <= n {
            if n % i == 0 {
                return false;
            }
            i += 2;
        }
        true
    }

    /// Discovery test: brute-force search small odd primes p > 2 for
    /// the first one making f̃_16 irreducible. Marked `#[ignore]`.
    #[test]
    #[ignore]
    fn find_irreducible_prime_scan() {
        // Small primes first.
        let mut found: Vec<u64> = Vec::new();
        for p in 3u64..=1_000 {
            if !is_prime(p) {
                continue;
            }
            if f_tilde_is_irreducible_mod_p_runtime(p) {
                found.push(p);
            }
        }
        println!("Small primes where f̃_16 is irreducible: {:?}", found);

        // Scan downward from just below 2^31, gather a few candidates.
        let mut big_found: Vec<u64> = Vec::new();
        let mut p: u64 = (1u64 << 31) - 1;
        while big_found.len() < 5 && p > (1u64 << 31) - 100_000_000 {
            if p % 2 == 1 && is_prime(p) && f_tilde_is_irreducible_mod_p_runtime(p) {
                println!("Big prime irreducible: p = {}", p);
                big_found.push(p);
            }
            p -= 1;
            if p == 0 { break; }
        }
        println!("Big primes where f̃_16 is irreducible: {:?}", big_found);
    }

    /// Runtime version of f_tilde_is_irreducible_mod_p (mirrors the
    /// const-generic version in `Gf2_16Ext::f_tilde_is_irreducible_mod_p`).
    fn f_tilde_is_irreducible_mod_p_runtime(p: u64) -> bool {
        // X represented as length-D coefficient array.
        let mut x = [0u64; D_16];
        x[1] = 1;

        // Compute X^{p^16} mod f̃ and X^{p^8} mod f̃ via repeated `X -> X^p`.
        let xp8 = x_to_p_power_k_runtime(&x, 8, p);
        let xp16 = x_to_p_power_k_runtime(&xp8, 8, p);
        // Check (a) X^{p^16} ≡ X.
        if xp16 != x {
            return false;
        }
        // Check (b) gcd(f̃, X^{p^8} - X) = 1.
        let mut diff = xp8;
        if diff[1] == 0 {
            diff[1] = p - 1;
        } else {
            diff[1] -= 1;
        }
        let mut diff_vec: Vec<u64> = diff.to_vec();
        while diff_vec.len() > 1 && *diff_vec.last().unwrap() == 0 {
            diff_vec.pop();
        }
        if diff_vec == [0] {
            return false;
        }
        let mut f_tilde = vec![0u64; D_16 + 1];
        f_tilde[D_16] = 1;
        for &k in &F_TILDE_LOWER_DEGREES_16 {
            f_tilde[k] = 1;
        }
        let g = super::poly_gcd(&f_tilde, &diff_vec, p);
        g.len() == 1 && g[0] != 0
    }

    fn mul_inner_runtime(a: &[u64; D_16], b: &[u64; D_16], p: u64) -> [u64; D_16] {
        let mut prod = [0u64; 2 * D_16 - 1];
        for i in 0..D_16 {
            let ai = a[i];
            if ai == 0 {
                continue;
            }
            for j in 0..D_16 {
                let bj = b[j];
                if bj == 0 {
                    continue;
                }
                prod[i + j] = super::add_mod_p(prod[i + j], super::mul_mod_p(ai, bj, p), p);
            }
        }
        for j in (D_16..(2 * D_16 - 1)).rev() {
            let c = prod[j];
            if c == 0 {
                continue;
            }
            prod[j] = 0;
            let base = j - D_16;
            for &shift in &F_TILDE_LOWER_DEGREES_16 {
                prod[base + shift] = super::add_mod_p(prod[base + shift], c, p);
            }
        }
        let mut out = [0u64; D_16];
        out.copy_from_slice(&prod[..D_16]);
        out
    }

    fn x_to_p_power_k_runtime(x: &[u64; D_16], k: u32, p: u64) -> [u64; D_16] {
        let mut acc = *x;
        for _ in 0..k {
            let mut result = [0u64; D_16];
            result[0] = 1;
            let mut base = acc;
            let mut exp = p;
            while exp > 0 {
                if exp & 1 == 1 {
                    result = mul_inner_runtime(&result, &base, p);
                }
                exp >>= 1;
                if exp > 0 {
                    base = mul_inner_runtime(&base, &base, p);
                }
            }
            acc = result;
        }
        acc
    }

    /// Sanity-check: confirm P_16_DEFAULT really does make f̃_16 irreducible
    /// (so `Gf2_16Ext<P_16_DEFAULT>` is the field F_{p^16}).
    #[test]
    fn default_prime_makes_f_tilde_irreducible() {
        assert!(
            Gf2_16Ext::<P_16_DEFAULT>::f_tilde_is_irreducible_mod_p(),
            "f̃_16 must be irreducible mod P_16_DEFAULT = {}",
            P_16_DEFAULT
        );
    }

    #[test]
    fn arithmetic_smoke() {
        const P: u64 = P_16_DEFAULT;
        let a = Gf2_16Ext::<P>::new_array({
            let mut c = [0u64; D_16];
            c[0] = 5;
            c[1] = 7;
            c[15] = 12345;
            c
        });
        let b = Gf2_16Ext::<P>::new_array({
            let mut c = [0u64; D_16];
            c[0] = 11;
            c[2] = 3;
            c[14] = 999;
            c
        });
        let sum = a + b;
        assert_eq!(sum.coeffs()[0], 16);
        assert_eq!(sum.coeffs()[1], 7);
        assert_eq!(sum.coeffs()[2], 3);
        let prod = a * b;
        assert!(!prod.is_zero());
    }
}
