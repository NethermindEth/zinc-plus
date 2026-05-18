//! Verifier-side scalar ring `F = F[X] / f̃` with `f̃ = X^16 + X^5 +
//! X^3 + X^2 + 1`, where the coefficient field `F` is
//! `crypto_primitives::crypto_bigint_monty::MontyField<LIMBS>` — a
//! dynamic-modulus prime field (256-bit-class when `LIMBS = 4`).
//!
//! This is the integrated-protocol counterpart of [`super::ext_field`]'s
//! `Gf2_16Ext<P>`: the *same ring shape* (`F[X]/f̃`, with the `f̃`-fold
//! `X^16 ≡ X^5 + X^3 + X^2 + 1`, lower degrees `{0, 2, 3, 5}`), but the
//! coefficients are `MontyField<LIMBS>` instead of `u64`-mod-`P`. The
//! Zinc+ protocol's prime is a 256-bit `MontyField`, which a
//! `u64`-coefficient `Gf2_16Ext` cannot hold, so `PolyExt16<LIMBS>` is
//! the binary lane's PCS field `F`.
//!
//! ## `f̃` is used only as a ring quotient
//!
//! `f̃` is **NOT** required to be irreducible over the coefficient field
//! — `PolyExt16` is used purely as a *ring*. Division / inversion are
//! never needed on the protocol's binary-lane path: the evaluation
//! point is scalar-valued, so the verifier never multiplies two
//! non-scalar `PolyExt16` elements. `Inv` / `Div` are therefore
//! implemented as an extended-Euclidean attempt that returns `None`
//! (`Inv`) or panics (`Div`) when `f̃` is not invertible at the given
//! element. They are *not* exercised by the protocol.
//!
//! ## The key difference from `Gf2_16Ext`: the runtime config
//!
//! `Gf2_16Ext` has `type Config = ()` because its prime `P` is a const
//! generic. `MontyField<LIMBS>` has a non-trivial runtime `Config` (its
//! modulus, `MontyParams<LIMBS>`). `PolyExt16<LIMBS>` therefore:
//!
//! - stores its own `Field::Inner` (a `PolyExt16Inner<LIMBS>`: the 16
//!   coefficients as Montgomery-form `Uint`s) **plus** the shared
//!   modulus config `MontyParams<LIMBS>`. Because the storage *is* the
//!   `Inner`, `Field::inner()` / `inner_mut()` are real, panic-free
//!   borrows;
//! - threads `Self::Config = <MontyField<LIMBS> as PrimeField>::Config`
//!   through every `*_with_cfg` method;
//! - delegates all arithmetic to `MontyField`'s own arithmetic: each
//!   coefficient is reconstructed on demand as a `MontyField` from its
//!   stored Montgomery-form `Uint` and the shared params via
//!   `MontyField::new_unchecked_with_cfg`, the op is performed in
//!   `MontyField`, and the result is written back with
//!   `MontyField::into_inner()`. The runtime modulus is always
//!   respected.
//!
//! Because the zero/one element needs the runtime config, `ConstZero` /
//! `ConstOne` are **not** implemented (they are impossible without a
//! compile-time config); see the crate-level integration notes. `Zero`
//! / `One` from `num_traits` are likewise not implemented (their
//! signatures `fn zero()` / `fn one()` take no config). The `*_with_cfg`
//! constructors from `PrimeField` cover the protocol's needs.

use core::{
    cmp::Ordering,
    fmt::{self, Debug, Display, Formatter},
    hash::{Hash, Hasher},
    iter::{Product, Sum},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use crypto_bigint::modular::MontyParams;
use crypto_primitives::{
    Field, FromWithConfig, PrimeField, Ring, Semiring, crypto_bigint_monty::MontyField,
    crypto_bigint_uint::Uint,
};
use num_traits::{CheckedAdd, CheckedMul, CheckedNeg, CheckedSub, ConstZero, Inv, Pow};
use zinc_poly::univariate::{binary::BinaryPoly, dense::DensePolynomial};
use zinc_transcript::traits::{ConstTranscribable, GenTranscribable};
use zinc_utils::{from_ref::FromRef, mul_by_scalar::MulByScalar};

use super::basis::{D_16, F_TILDE_LOWER_DEGREES_16};
use super::ext_field::Bit4Poly16;
use super::packed_cw::PackedI64Poly16;

/// Coefficient field of `PolyExt16<LIMBS>`.
type Coef<const LIMBS: usize> = MontyField<LIMBS>;
/// Config type threaded through every `*_with_cfg` method.
type Cfg<const LIMBS: usize> = <MontyField<LIMBS> as PrimeField>::Config;

// ===========================================================================
// PolyExt16Inner — newtype over [MontyField::Inner; 16].
//
// `MontyField<LIMBS>::Inner = Uint<LIMBS>`, which already implements
// `GenTranscribable` / `ConstTranscribable`. Wrapping the array in a
// newtype keeps the transcript impls clear of the orphan rule.
// ===========================================================================

/// Inner representation of a `PolyExt16<LIMBS>`: the 16 coefficients in
/// their `MontyField` *inner* (Montgomery) form. Used as
/// `<PolyExt16 as Field>::Inner`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct PolyExt16Inner<const LIMBS: usize>(pub [Uint<LIMBS>; D_16]);

impl<const LIMBS: usize> PolyExt16Inner<LIMBS> {
    /// All-zero inner (every coefficient's Montgomery form is 0, which
    /// `MontyForm` also uses for the field element 0).
    fn zeros() -> Self {
        Self(core::array::from_fn(|_| Uint::<LIMBS>::ZERO))
    }
}

impl<const LIMBS: usize> Default for PolyExt16Inner<LIMBS> {
    fn default() -> Self {
        Self::zeros()
    }
}

impl<const LIMBS: usize> GenTranscribable for PolyExt16Inner<LIMBS> {
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        let w = <Uint<LIMBS> as ConstTranscribable>::NUM_BYTES;
        let coeffs = core::array::from_fn(|i| {
            <Uint<LIMBS> as GenTranscribable>::read_transcription_bytes_exact(
                &bytes[i * w..(i + 1) * w],
            )
        });
        Self(coeffs)
    }

    fn write_transcription_bytes_exact(&self, buf: &mut [u8]) {
        let w = <Uint<LIMBS> as ConstTranscribable>::NUM_BYTES;
        for (i, limb) in self.0.iter().enumerate() {
            limb.write_transcription_bytes_exact(&mut buf[i * w..(i + 1) * w]);
        }
    }
}

impl<const LIMBS: usize> ConstTranscribable for PolyExt16Inner<LIMBS> {
    const NUM_BYTES: usize = D_16 * <Uint<LIMBS> as ConstTranscribable>::NUM_BYTES;
    const NUM_BITS: usize = D_16 * <Uint<LIMBS> as ConstTranscribable>::NUM_BITS;
}

// ===========================================================================
// PolyExt16Modulus — the coefficient modulus, width-matched to `Inner`.
//
// `PrimeField::Modulus` is used by the protocol's PCS transcript
// (`pcs_transcript::write_field_element_no_length`), which reuses a
// SINGLE buffer sized at `Inner::NUM_BYTES` for both the modulus and the
// inner writes. So `Modulus::NUM_BYTES` must equal `Inner::NUM_BYTES`
// (= `D_16 * Uint<LIMBS>::NUM_BYTES`).
//
// `MontyField<LIMBS>`'s own modulus is just a `Uint<LIMBS>` (one
// coefficient's worth), which is `D_16`× too narrow. `PolyExt16Modulus`
// is the width-matched newtype: it stores the single coefficient
// modulus and transcribes it into slot 0 of a `D_16`-slot buffer,
// zero-padding the remaining slots.
// ===========================================================================

/// The modulus of `PolyExt16<LIMBS>`: the coefficient field's modulus
/// (`Uint<LIMBS>`), width-matched to [`PolyExt16Inner`] for transcript
/// purposes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PolyExt16Modulus<const LIMBS: usize>(pub Uint<LIMBS>);

impl<const LIMBS: usize> GenTranscribable for PolyExt16Modulus<LIMBS> {
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        let w = <Uint<LIMBS> as ConstTranscribable>::NUM_BYTES;
        // The coefficient modulus lives in slot 0; the remaining
        // `D_16 - 1` zero-padded slots are ignored.
        Self(<Uint<LIMBS> as GenTranscribable>::read_transcription_bytes_exact(
            &bytes[..w],
        ))
    }

    fn write_transcription_bytes_exact(&self, buf: &mut [u8]) {
        buf.fill(0);
        let w = <Uint<LIMBS> as ConstTranscribable>::NUM_BYTES;
        self.0.write_transcription_bytes_exact(&mut buf[..w]);
    }
}

impl<const LIMBS: usize> ConstTranscribable for PolyExt16Modulus<LIMBS> {
    const NUM_BYTES: usize = D_16 * <Uint<LIMBS> as ConstTranscribable>::NUM_BYTES;
    const NUM_BITS: usize = D_16 * <Uint<LIMBS> as ConstTranscribable>::NUM_BITS;
}

/// `Uint<LIMBS>` (the protocol's `Fmod`) → `PolyExt16Modulus`. Required
/// by `verify_folded`'s `BinF::Modulus: FromRef<ZtF::Fmod>` bound.
impl<const LIMBS: usize> FromRef<Uint<LIMBS>> for PolyExt16Modulus<LIMBS> {
    fn from_ref(value: &Uint<LIMBS>) -> Self {
        Self(*value)
    }
}

impl<const LIMBS: usize> FromRef<PolyExt16Modulus<LIMBS>> for PolyExt16Modulus<LIMBS> {
    fn from_ref(value: &Self) -> Self {
        *value
    }
}

// ===========================================================================
// PolyExt16<LIMBS> — F[X] / f̃ with F = MontyField<LIMBS>.
// ===========================================================================

/// The ring `F[X] / f̃`, `f̃ = X^16 + X^5 + X^3 + X^2 + 1`, with
/// coefficient field `F = MontyField<LIMBS>`.
///
/// The storage **is** the `Field::Inner`: a [`PolyExt16Inner`] holding
/// the 16 coefficients in `MontyField` *inner* (Montgomery) form, plus
/// the shared modulus config. All 16 coefficients carry the same
/// config.
#[derive(Clone)]
pub struct PolyExt16<const LIMBS: usize> {
    /// The 16 coefficients in Montgomery form, `inner.0[i]` is the
    /// coefficient of `X^i`. This is exactly `Field::Inner`.
    inner: PolyExt16Inner<LIMBS>,
    /// The shared modulus config for every coefficient.
    params: MontyParams<LIMBS>,
}

impl<const LIMBS: usize> PolyExt16<LIMBS> {
    /// Reconstruct coefficient `i` as a `MontyField`. The stored `Uint`
    /// is already in `MontyField::inner()` (Montgomery) form, so
    /// `new_unchecked_with_cfg` is the correct, cheap reconstruction —
    /// `new_with_cfg` would wrongly re-reduce it.
    #[inline]
    fn coef(&self, i: usize) -> Coef<LIMBS> {
        Coef::<LIMBS>::new_unchecked_with_cfg(self.inner.0[i], &self.params)
    }

    /// All 16 coefficients reconstructed as `MontyField`s, low degree
    /// first.
    #[inline]
    fn coef_array(&self) -> [Coef<LIMBS>; D_16] {
        core::array::from_fn(|i| self.coef(i))
    }

    /// Build a `PolyExt16` from 16 `MontyField` coefficients sharing
    /// `params`, storing each in its `into_inner()` (Montgomery) form.
    #[inline]
    fn from_coef_array(coeffs: [Coef<LIMBS>; D_16], params: MontyParams<LIMBS>) -> Self {
        Self {
            inner: PolyExt16Inner(coeffs.map(Coef::<LIMBS>::into_inner)),
            params,
        }
    }

    /// Builds an element from 16 coefficients (which must all share the
    /// same modulus config).
    #[inline]
    pub fn from_coeffs(coeffs: [Coef<LIMBS>; D_16]) -> Self {
        let params = *coeffs[0].cfg();
        Self::from_coef_array(coeffs, params)
    }

    /// The 16 coefficients, low degree first.
    #[inline]
    pub fn coeffs(&self) -> [Coef<LIMBS>; D_16] {
        self.coef_array()
    }

    /// The shared modulus config.
    #[inline]
    fn config(&self) -> &Cfg<LIMBS> {
        &self.params
    }

    /// Zero element under the given config.
    pub fn zero_with(cfg: &Cfg<LIMBS>) -> Self {
        Self {
            inner: PolyExt16Inner::zeros(),
            params: *cfg,
        }
    }

    /// One element (`1·X^0`) under the given config.
    pub fn one_with(cfg: &Cfg<LIMBS>) -> Self {
        let mut coeffs: [Coef<LIMBS>; D_16] =
            core::array::from_fn(|_| Coef::<LIMBS>::zero_with_cfg(cfg));
        coeffs[0] = Coef::<LIMBS>::one_with_cfg(cfg);
        Self::from_coef_array(coeffs, *cfg)
    }

    /// Builds the constant polynomial `c·X^0` from a scalar coefficient.
    pub fn from_scalar(c: Coef<LIMBS>) -> Self {
        let cfg = *c.cfg();
        let mut coeffs: [Coef<LIMBS>; D_16] =
            core::array::from_fn(|_| Coef::<LIMBS>::zero_with_cfg(&cfg));
        coeffs[0] = c;
        Self::from_coef_array(coeffs, cfg)
    }

    /// `true` iff every coefficient is zero. The all-zero Montgomery
    /// form is exactly the field element 0, so this is a pure `Uint`
    /// check — no reconstruction needed.
    pub fn is_zero(&self) -> bool {
        self.inner.0.iter().all(|u| u == &Uint::<LIMBS>::ZERO)
    }

    /// Schoolbook poly-mult of two length-16 coefficient arrays, then
    /// reduce mod `f̃` via `X^16 ≡ X^5 + X^3 + X^2 + 1` (lower degrees
    /// `{0, 2, 3, 5}`). Every coefficient operation is a `MontyField`
    /// operation, so the runtime modulus is respected automatically.
    fn mul_inner(a: &[Coef<LIMBS>; D_16], b: &[Coef<LIMBS>; D_16]) -> Self {
        let cfg = *a[0].cfg();
        // Length-(2D-1) product buffer, all coefficients zero-initialised.
        let mut prod: [Coef<LIMBS>; 2 * D_16 - 1] =
            core::array::from_fn(|_| Coef::<LIMBS>::zero_with_cfg(&cfg));
        for (i, ai) in a.iter().enumerate() {
            if Coef::<LIMBS>::is_zero(ai) {
                continue;
            }
            for (j, bj) in b.iter().enumerate() {
                if Coef::<LIMBS>::is_zero(bj) {
                    continue;
                }
                prod[i + j] = &prod[i + j] + &(ai * bj);
            }
        }
        // Fold high monomials X^j (j ≥ 16) down using f̃.
        for j in (D_16..(2 * D_16 - 1)).rev() {
            if Coef::<LIMBS>::is_zero(&prod[j]) {
                continue;
            }
            let c = prod[j].clone();
            prod[j] = Coef::<LIMBS>::zero_with_cfg(&cfg);
            let base = j - D_16;
            for &shift in &F_TILDE_LOWER_DEGREES_16 {
                prod[base + shift] = &prod[base + shift] + &c;
            }
        }
        let coeffs: [Coef<LIMBS>; D_16] = core::array::from_fn(|i| prod[i].clone());
        Self::from_coef_array(coeffs, cfg)
    }

    /// Extended-Euclidean inversion attempt in `F[X]` modulo `f̃`.
    ///
    /// Returns `Some(inv)` only when `gcd(self, f̃)` is a non-zero
    /// constant *and* `f̃`'s leading-coefficient inversion succeeds.
    /// Because `f̃` need not be irreducible over `MontyField<LIMBS>`,
    /// this returns `None` for many inputs — which is acceptable: the
    /// protocol's binary lane never inverts a `PolyExt16` element. See
    /// the module docs.
    pub fn try_inv(&self) -> Option<Self> {
        let cfg = *self.config();
        let zero = Coef::<LIMBS>::zero_with_cfg(&cfg);
        let one = Coef::<LIMBS>::one_with_cfg(&cfg);

        // f̃ as a length-17 coefficient vector.
        let mut f_tilde = vec![zero.clone(); D_16 + 1];
        f_tilde[D_16] = one.clone();
        for &k in &F_TILDE_LOWER_DEGREES_16 {
            f_tilde[k] = one.clone();
        }

        let mut r0 = f_tilde;
        let mut r1: Vec<Coef<LIMBS>> = self.coef_array().to_vec();
        trim(&mut r1, &zero);
        if r1.iter().all(Coef::<LIMBS>::is_zero) {
            return None;
        }
        let mut s0: Vec<Coef<LIMBS>> = vec![zero.clone()];
        let mut s1: Vec<Coef<LIMBS>> = vec![one.clone()];

        while !(r1.len() == 1 && Coef::<LIMBS>::is_zero(&r1[0])) {
            let (q, rem) = poly_divmod(&r0, &r1, &cfg)?;
            let qs1 = poly_mul(&q, &s1, &cfg);
            let new_s = poly_sub(&s0, &qs1, &cfg);
            r0 = r1;
            r1 = rem;
            s0 = s1;
            s1 = new_s;
        }
        // gcd lives in r0; it must be a non-zero constant for an inverse.
        if r0.len() != 1 || Coef::<LIMBS>::is_zero(&r0[0]) {
            return None;
        }
        let inv_g = Inv::inv(r0[0].clone())?;
        let mut coeffs: [Coef<LIMBS>; D_16] =
            core::array::from_fn(|_| Coef::<LIMBS>::zero_with_cfg(&cfg));
        for (i, c) in s0.iter().enumerate().take(D_16) {
            coeffs[i] = c * &inv_g;
        }
        Some(Self::from_coef_array(coeffs, cfg))
    }
}

// --- F[X] dense-vector helpers (used only by `try_inv`) --------------------

/// Drop trailing zero coefficients, keeping at least one element.
fn trim<const LIMBS: usize>(v: &mut Vec<Coef<LIMBS>>, zero: &Coef<LIMBS>) {
    while v.len() > 1 && Coef::<LIMBS>::is_zero(v.last().unwrap()) {
        v.pop();
    }
    if v.is_empty() {
        v.push(zero.clone());
    }
}

fn poly_sub<const LIMBS: usize>(
    a: &[Coef<LIMBS>],
    b: &[Coef<LIMBS>],
    cfg: &Cfg<LIMBS>,
) -> Vec<Coef<LIMBS>> {
    let zero = Coef::<LIMBS>::zero_with_cfg(cfg);
    let n = a.len().max(b.len());
    let mut out = vec![zero.clone(); n];
    for (i, slot) in out.iter_mut().enumerate() {
        let ai = a.get(i).cloned().unwrap_or_else(|| zero.clone());
        let bi = b.get(i).cloned().unwrap_or_else(|| zero.clone());
        *slot = ai - bi;
    }
    trim(&mut out, &zero);
    out
}

fn poly_mul<const LIMBS: usize>(
    a: &[Coef<LIMBS>],
    b: &[Coef<LIMBS>],
    cfg: &Cfg<LIMBS>,
) -> Vec<Coef<LIMBS>> {
    let zero = Coef::<LIMBS>::zero_with_cfg(cfg);
    if a.is_empty() || b.is_empty() {
        return vec![zero];
    }
    let mut out = vec![zero.clone(); a.len() + b.len() - 1];
    for (i, ai) in a.iter().enumerate() {
        if Coef::<LIMBS>::is_zero(ai) {
            continue;
        }
        for (j, bj) in b.iter().enumerate() {
            out[i + j] = &out[i + j] + &(ai * bj);
        }
    }
    out
}

/// Polynomial division with remainder in `F[X]`. Returns `None` when the
/// divisor's leading coefficient is not invertible (possible because the
/// coefficient field's modulus is dynamic but always prime, so this only
/// happens for a zero divisor — already excluded by callers).
fn poly_divmod<const LIMBS: usize>(
    num: &[Coef<LIMBS>],
    denom: &[Coef<LIMBS>],
    cfg: &Cfg<LIMBS>,
) -> Option<(Vec<Coef<LIMBS>>, Vec<Coef<LIMBS>>)> {
    let zero = Coef::<LIMBS>::zero_with_cfg(cfg);
    let mut denom = denom.to_vec();
    trim(&mut denom, &zero);
    let mut rem = num.to_vec();
    trim(&mut rem, &zero);
    if rem.len() < denom.len() {
        return Some((vec![zero], rem));
    }
    let lead_denom_inv = Inv::inv(denom.last().unwrap().clone())?;
    let mut q = vec![zero.clone(); rem.len() - denom.len() + 1];
    while rem.len() >= denom.len() && !(rem.len() == 1 && Coef::<LIMBS>::is_zero(&rem[0])) {
        let lead = rem.last().unwrap().clone();
        if Coef::<LIMBS>::is_zero(&lead) {
            rem.pop();
            continue;
        }
        let coeff = lead * &lead_denom_inv;
        let pos = rem.len() - denom.len();
        for (i, di) in denom.iter().enumerate() {
            rem[pos + i] = &rem[pos + i] - &(&coeff * di);
        }
        q[pos] = coeff;
        trim(&mut rem, &zero);
    }
    Some((q, rem))
}

// ===========================================================================
// Core trait impls.
// ===========================================================================

impl<const LIMBS: usize> PartialEq for PolyExt16<LIMBS> {
    fn eq(&self, other: &Self) -> bool {
        // Equality of the stored Montgomery-form coefficients. (The
        // config is not compared, matching the prior coefficient-wise
        // `MontyField` equality, which also ignores params.)
        self.inner == other.inner
    }
}
impl<const LIMBS: usize> Eq for PolyExt16<LIMBS> {}

impl<const LIMBS: usize> Hash for PolyExt16<LIMBS> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

impl<const LIMBS: usize> Debug for PolyExt16<LIMBS> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "PolyExt16<{LIMBS}>({:?})", &self.coef_array())
    }
}

impl<const LIMBS: usize> Display for PolyExt16<LIMBS> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "PolyExt16<{LIMBS}>(...)")
    }
}

impl<const LIMBS: usize> PartialOrd for PolyExt16<LIMBS> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Lexicographic over the coefficients' Montgomery forms.
        for (a, b) in self.inner.0.iter().zip(other.inner.0.iter()) {
            match a.cmp(b) {
                Ordering::Equal => continue,
                non_eq => return Some(non_eq),
            }
        }
        Some(Ordering::Equal)
    }
}

// ===========================================================================
// Arithmetic operators, delegated coefficient-wise to MontyField.
// ===========================================================================

impl<const LIMBS: usize> Neg for PolyExt16<LIMBS> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        let params = self.params;
        let coeffs = self.coef_array().map(|c| -c);
        Self::from_coef_array(coeffs, params)
    }
}

impl<const LIMBS: usize> AddAssign<&Self> for PolyExt16<LIMBS> {
    fn add_assign(&mut self, rhs: &Self) {
        let params = self.params;
        for i in 0..D_16 {
            let a = Coef::<LIMBS>::new_unchecked_with_cfg(self.inner.0[i], &params);
            let b = Coef::<LIMBS>::new_unchecked_with_cfg(rhs.inner.0[i], &params);
            self.inner.0[i] = (a + &b).into_inner();
        }
    }
}
impl<const LIMBS: usize> AddAssign for PolyExt16<LIMBS> {
    fn add_assign(&mut self, rhs: Self) {
        *self += &rhs;
    }
}
impl<const LIMBS: usize> Add for PolyExt16<LIMBS> {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self {
        self += &rhs;
        self
    }
}
impl<const LIMBS: usize> Add<&Self> for PolyExt16<LIMBS> {
    type Output = Self;
    fn add(mut self, rhs: &Self) -> Self {
        self += rhs;
        self
    }
}

impl<const LIMBS: usize> SubAssign<&Self> for PolyExt16<LIMBS> {
    fn sub_assign(&mut self, rhs: &Self) {
        let params = self.params;
        for i in 0..D_16 {
            let a = Coef::<LIMBS>::new_unchecked_with_cfg(self.inner.0[i], &params);
            let b = Coef::<LIMBS>::new_unchecked_with_cfg(rhs.inner.0[i], &params);
            self.inner.0[i] = (a - &b).into_inner();
        }
    }
}
impl<const LIMBS: usize> SubAssign for PolyExt16<LIMBS> {
    fn sub_assign(&mut self, rhs: Self) {
        *self -= &rhs;
    }
}
impl<const LIMBS: usize> Sub for PolyExt16<LIMBS> {
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self {
        self -= &rhs;
        self
    }
}
impl<const LIMBS: usize> Sub<&Self> for PolyExt16<LIMBS> {
    type Output = Self;
    fn sub(mut self, rhs: &Self) -> Self {
        self -= rhs;
        self
    }
}

impl<const LIMBS: usize> Mul for PolyExt16<LIMBS> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::mul_inner(&self.coef_array(), &rhs.coef_array())
    }
}
impl<const LIMBS: usize> Mul<&Self> for PolyExt16<LIMBS> {
    type Output = Self;
    fn mul(self, rhs: &Self) -> Self {
        Self::mul_inner(&self.coef_array(), &rhs.coef_array())
    }
}
impl<const LIMBS: usize> MulAssign for PolyExt16<LIMBS> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = Self::mul_inner(&self.coef_array(), &rhs.coef_array());
    }
}
impl<const LIMBS: usize> MulAssign<&Self> for PolyExt16<LIMBS> {
    fn mul_assign(&mut self, rhs: &Self) {
        *self = Self::mul_inner(&self.coef_array(), &rhs.coef_array());
    }
}

impl<const LIMBS: usize> Div for PolyExt16<LIMBS> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let inv = rhs
            .try_inv()
            .expect("PolyExt16::div: divisor not invertible mod f̃");
        self * inv
    }
}
impl<const LIMBS: usize> Div<&Self> for PolyExt16<LIMBS> {
    type Output = Self;
    fn div(self, rhs: &Self) -> Self {
        let inv = rhs
            .try_inv()
            .expect("PolyExt16::div: divisor not invertible mod f̃");
        self * inv
    }
}
impl<const LIMBS: usize> DivAssign for PolyExt16<LIMBS> {
    fn div_assign(&mut self, rhs: Self) {
        *self = self.clone() / rhs;
    }
}
impl<const LIMBS: usize> DivAssign<&Self> for PolyExt16<LIMBS> {
    fn div_assign(&mut self, rhs: &Self) {
        *self = self.clone() / rhs;
    }
}

impl<const LIMBS: usize> Pow<u32> for PolyExt16<LIMBS> {
    type Output = Self;
    fn pow(self, mut exp: u32) -> Self {
        let mut result = Self::one_with(self.config());
        let mut base = self;
        while exp > 0 {
            if exp & 1 == 1 {
                result = Self::mul_inner(&result.coef_array(), &base.coef_array());
            }
            exp >>= 1;
            if exp > 0 {
                base = Self::mul_inner(&base.coef_array(), &base.coef_array());
            }
        }
        result
    }
}

impl<const LIMBS: usize> Inv for PolyExt16<LIMBS> {
    type Output = Option<Self>;
    fn inv(self) -> Self::Output {
        self.try_inv()
    }
}

// --- num-traits checked ops (field ops never overflow) ---------------------

impl<const LIMBS: usize> CheckedAdd for PolyExt16<LIMBS> {
    fn checked_add(&self, rhs: &Self) -> Option<Self> {
        Some(self.clone() + rhs)
    }
}
impl<const LIMBS: usize> CheckedSub for PolyExt16<LIMBS> {
    fn checked_sub(&self, rhs: &Self) -> Option<Self> {
        Some(self.clone() - rhs)
    }
}
impl<const LIMBS: usize> CheckedMul for PolyExt16<LIMBS> {
    fn checked_mul(&self, rhs: &Self) -> Option<Self> {
        Some(self.clone() * rhs)
    }
}
impl<const LIMBS: usize> CheckedNeg for PolyExt16<LIMBS> {
    fn checked_neg(&self) -> Option<Self> {
        Some(-self.clone())
    }
}

// --- Sum / Product. These panic on an empty iterator because there is
//     no config to build a zero/one from — exactly like `MontyField`. ---

impl<const LIMBS: usize> Sum for PolyExt16<LIMBS> {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let first = iter
            .next()
            .expect("Sum of an empty iterator is not defined for PolyExt16");
        iter.fold(first, |acc, x| acc + x)
    }
}
impl<'a, const LIMBS: usize> Sum<&'a Self> for PolyExt16<LIMBS> {
    fn sum<I: Iterator<Item = &'a Self>>(mut iter: I) -> Self {
        let first = iter
            .next()
            .expect("Sum of an empty iterator is not defined for PolyExt16")
            .clone();
        iter.fold(first, |acc, x| acc + x)
    }
}
impl<const LIMBS: usize> Product for PolyExt16<LIMBS> {
    fn product<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let first = iter
            .next()
            .expect("Product of an empty iterator is not defined for PolyExt16");
        iter.fold(first, |acc, x| acc * x)
    }
}
impl<'a, const LIMBS: usize> Product<&'a Self> for PolyExt16<LIMBS> {
    fn product<I: Iterator<Item = &'a Self>>(mut iter: I) -> Self {
        let first = iter
            .next()
            .expect("Product of an empty iterator is not defined for PolyExt16")
            .clone();
        iter.fold(first, |acc, x| acc * x)
    }
}

// ===========================================================================
// Semiring / Ring / Field / PrimeField.
//
// NOTE: `ConstZero` / `ConstOne` and num-traits `Zero` / `One` are NOT
// implemented: the zero/one element needs the runtime modulus config,
// which those traits' const / no-arg signatures cannot supply. The
// `PrimeField::{zero,one}_with_cfg` constructors cover the protocol's
// needs. This is the one structural deviation from `Gf2_16Ext`, forced
// by `MontyField`'s dynamic config. `Semiring` / `Ring` do not require
// `Zero` / `One` (only `FixedSemiring` does), so the trait stack still
// closes.
// ===========================================================================

impl<const LIMBS: usize> Semiring for PolyExt16<LIMBS> {}
impl<const LIMBS: usize> Ring for PolyExt16<LIMBS> {}

impl<const LIMBS: usize> Field for PolyExt16<LIMBS> {
    type Inner = PolyExt16Inner<LIMBS>;
    // Width-matched to `Inner` (`D_16 * Uint<LIMBS>::NUM_BYTES`): the
    // protocol's PCS transcript reuses one `Inner`-sized buffer for both
    // the modulus and inner writes. `MontyField`'s bare `Uint<LIMBS>`
    // modulus is `D_16`× too narrow; `PolyExt16Modulus` wraps it with
    // the matching transcript width.
    type Modulus = PolyExt16Modulus<LIMBS>;

    fn inner(&self) -> &Self::Inner {
        // The storage *is* the `Inner`: a real, panic-free borrow.
        &self.inner
    }

    fn inner_mut(&mut self) -> &mut Self::Inner {
        // The storage *is* the `Inner`: a real, panic-free borrow.
        &mut self.inner
    }

    fn into_inner(self) -> Self::Inner {
        self.inner
    }
}

impl<const LIMBS: usize> PrimeField for PolyExt16<LIMBS> {
    type Config = Cfg<LIMBS>;

    fn cfg(&self) -> &Self::Config {
        self.config()
    }

    fn is_zero(value: &Self) -> bool {
        value.is_zero()
    }

    fn modulus(&self) -> Self::Modulus {
        // The coefficient field's modulus; `f̃` is a *ring* quotient and
        // is not part of the modulus the protocol transcribes. Wrapped
        // in `PolyExt16Modulus` so its transcript width matches `Inner`.
        PolyExt16Modulus(self.coef(0).modulus())
    }

    fn modulus_minus_one_div_two(&self) -> Self::Inner {
        // `f̃` is used only as a ring; there is no Legendre-symbol path
        // for `PolyExt16`. Mirror `Gf2_16Ext`, which returns an all-zero
        // inner here.
        PolyExt16Inner::zeros()
    }

    fn make_cfg(modulus: &Self::Modulus) -> Result<Self::Config, crypto_primitives::FieldError> {
        // Delegate to the coefficient field: the config IS the
        // coefficient field's config. Unwrap the width-matched newtype
        // to recover the bare `Uint<LIMBS>` coefficient modulus.
        <MontyField<LIMBS> as PrimeField>::make_cfg(&modulus.0)
    }

    fn new_with_cfg(inner: Self::Inner, cfg: &Self::Config) -> Self {
        // `new_with_cfg` treats each `Uint` as an ordinary integer and
        // reduces it into Montgomery form. Normalise through
        // `MontyField` so the stored `Inner` is canonical Montgomery
        // form (matching what `into_inner()` would produce).
        let coeffs = inner
            .0
            .map(|u| <MontyField<LIMBS> as PrimeField>::new_with_cfg(u, cfg));
        Self::from_coef_array(coeffs, *cfg)
    }

    fn new_unchecked_with_cfg(inner: Self::Inner, cfg: &Self::Config) -> Self {
        // The `Uint`s are already in `MontyField::inner()` (Montgomery)
        // form — exactly this struct's storage convention — so the
        // inner is stored verbatim with no reconstruction.
        Self {
            inner,
            params: *cfg,
        }
    }

    fn zero_with_cfg(cfg: &Self::Config) -> Self {
        Self::zero_with(cfg)
    }

    fn one_with_cfg(cfg: &Self::Config) -> Self {
        Self::one_with(cfg)
    }
}

// ===========================================================================
// FromRef / FromWithConfig conversions.
// ===========================================================================

impl<const LIMBS: usize> FromRef<PolyExt16<LIMBS>> for PolyExt16<LIMBS> {
    fn from_ref(value: &Self) -> Self {
        value.clone()
    }
}

impl<const LIMBS: usize> FromWithConfig<&PolyExt16<LIMBS>> for PolyExt16<LIMBS> {
    fn from_with_cfg(value: &Self, _cfg: &Self::Config) -> Self {
        value.clone()
    }
}

/// `BinaryPoly<16>` → `PolyExt16`: each `{0,1}` bit maps to the
/// `MontyField` 0 or 1.
impl<const LIMBS: usize> FromWithConfig<&BinaryPoly<16>> for PolyExt16<LIMBS> {
    fn from_with_cfg(value: &BinaryPoly<16>, cfg: &Self::Config) -> Self {
        let zero = Coef::<LIMBS>::zero_with_cfg(cfg);
        let one = Coef::<LIMBS>::one_with_cfg(cfg);
        let mut coeffs: [Coef<LIMBS>; D_16] = core::array::from_fn(|_| zero.clone());
        for (i, b) in value.iter().enumerate().take(D_16) {
            if b.into_inner() {
                coeffs[i] = one.clone();
            }
        }
        Self::from_coef_array(coeffs, *cfg)
    }
}

/// `DensePolynomial<i64, 16>` → `PolyExt16`: the `Cw` / `CombR`
/// projection. Each `i64` coefficient is reduced into `MontyField`
/// using `MontyField`'s signed `FromWithConfig<i64>` (correct signed
/// mod-p reduction).
impl<const LIMBS: usize> FromWithConfig<&DensePolynomial<i64, 16>> for PolyExt16<LIMBS> {
    fn from_with_cfg(value: &DensePolynomial<i64, 16>, cfg: &Self::Config) -> Self {
        let coeffs = value
            .coeffs
            .map(|c| <MontyField<LIMBS> as FromWithConfig<i64>>::from_with_cfg(c, cfg));
        Self::from_coef_array(coeffs, *cfg)
    }
}

/// `PackedI64Poly16<B>` → `PolyExt16`: the bit-packed `Cw` / `CombR`
/// projection; delegates to the `DensePolynomial<i64, 16>` impl.
impl<const LIMBS: usize, const B: u32> FromWithConfig<&PackedI64Poly16<B>> for PolyExt16<LIMBS> {
    fn from_with_cfg(value: &PackedI64Poly16<B>, cfg: &Self::Config) -> Self {
        <Self as FromWithConfig<&DensePolynomial<i64, 16>>>::from_with_cfg(&value.0, cfg)
    }
}

/// `Bit4Poly16` → `PolyExt16`: lifts the 16 nibbles (each in `0..16`)
/// into the extension as polynomial coefficients `0..16`. This is the
/// `Chal → F` lift the binary lane's coherence check `⟨coeffs, b⟩`
/// applies to each row-batching coefficient. It agrees with
/// `PackedI64Poly16: MulByScalar<&Bit4Poly16>` (the action that builds
/// `combined_row`): both read `Bit4Poly16` as the nibble polynomial in
/// `Z[X]/f̃`, and the codeword→F lift is a ring homomorphism, so the
/// coherence identity `⟨combined_row, q_1⟩ = ⟨coeffs, b⟩` holds.
/// `Bit4Poly16` is already degree `< 16`, so no mod-f̃ reduction is
/// needed.
impl<const LIMBS: usize> FromWithConfig<&Bit4Poly16> for PolyExt16<LIMBS> {
    fn from_with_cfg(value: &Bit4Poly16, cfg: &Self::Config) -> Self {
        let nibbles = value.nibbles();
        let coeffs: [Coef<LIMBS>; D_16] = core::array::from_fn(|i| {
            <MontyField<LIMBS> as FromWithConfig<u8>>::from_with_cfg(nibbles[i], cfg)
        });
        Self::from_coef_array(coeffs, *cfg)
    }
}

// ---------------------------------------------------------------------------
// Primitive-integer → constant-polynomial conversions.
//
// Each primitive scalar `v` maps to the degree-0 constant polynomial
// `v·X^0` (the scalar, reduced mod p, in coefficient 0; coefficients
// 1..16 zero). The scalar is reduced into the coefficient field via
// `MontyField`'s own signed/unsigned `FromWithConfig` (correct mod-p
// reduction). Implementing all ten primitive widths satisfies the
// blanket `FromPrimitiveWithConfig` impl in `crypto-primitives`.
// ---------------------------------------------------------------------------

/// Builds the constant polynomial `c·X^0` for a `MontyField` coefficient.
#[inline]
fn const_poly_from_coef<const LIMBS: usize>(
    c: Coef<LIMBS>,
    cfg: &Cfg<LIMBS>,
) -> PolyExt16<LIMBS> {
    let mut coeffs: [Coef<LIMBS>; D_16] =
        core::array::from_fn(|_| Coef::<LIMBS>::zero_with_cfg(cfg));
    coeffs[0] = c;
    PolyExt16::from_coef_array(coeffs, *cfg)
}

macro_rules! impl_from_primitive {
    ($($t:ty),* $(,)?) => {
        $(
            impl<const LIMBS: usize> FromWithConfig<$t> for PolyExt16<LIMBS> {
                fn from_with_cfg(value: $t, cfg: &Self::Config) -> Self {
                    let c = <MontyField<LIMBS> as FromWithConfig<$t>>::from_with_cfg(value, cfg);
                    const_poly_from_coef(c, cfg)
                }
            }

            impl<const LIMBS: usize> FromWithConfig<&$t> for PolyExt16<LIMBS> {
                fn from_with_cfg(value: &$t, cfg: &Self::Config) -> Self {
                    <Self as FromWithConfig<$t>>::from_with_cfg(*value, cfg)
                }
            }
        )*
    };
}

// All ten primitive integer widths — this makes `PolyExt16` satisfy the
// blanket `FromPrimitiveWithConfig` impl required by `verify_folded`'s
// `BinF` bounds.
impl_from_primitive!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128);

/// `MontyField<LIMBS>` scalar → `PolyExt16`: the constant polynomial
/// whose coefficient 0 is the scalar (rest zero). This is the `BinF:
/// FromWithConfig<&F>` embedding the protocol uses to lift the scalar
/// evaluation point `r0_ext` into the binary-lane PCS field.
impl<const LIMBS: usize> FromWithConfig<&MontyField<LIMBS>> for PolyExt16<LIMBS> {
    fn from_with_cfg(value: &MontyField<LIMBS>, _cfg: &Self::Config) -> Self {
        // `from_scalar` builds the degree-0 constant polynomial directly
        // from a `MontyField` coefficient (its config is the shared cfg).
        Self::from_scalar(value.clone())
    }
}

impl<const LIMBS: usize> FromWithConfig<MontyField<LIMBS>> for PolyExt16<LIMBS> {
    fn from_with_cfg(value: MontyField<LIMBS>, _cfg: &Self::Config) -> Self {
        Self::from_scalar(value)
    }
}

// ===========================================================================
// MulByScalar.
// ===========================================================================

impl<const LIMBS: usize> MulByScalar<&PolyExt16<LIMBS>> for PolyExt16<LIMBS> {
    fn mul_by_scalar<const CHECK: bool>(&self, rhs: &Self) -> Option<Self> {
        Some(self.clone() * rhs)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_bigint::Odd;
    use crypto_primitives::crypto_bigint_monty::MontyField;

    /// Concrete instantiation used by the tests: 256-bit coefficients.
    const LIMBS: usize = 4;
    type P = PolyExt16<LIMBS>;
    type F = MontyField<LIMBS>;
    type FCfg = <F as PrimeField>::Config;

    /// Build a `MontyField` config from a 256-bit test prime
    /// (secp256k1's field prime `2^256 - 2^32 - 977`, same prime used by
    /// `crypto-primitives`' own `MontyField` tests).
    fn test_cfg() -> FCfg {
        let modulus = crypto_bigint::Uint::<LIMBS>::from_be_hex(
            "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f",
        );
        let modulus = Odd::new(modulus).expect("modulus is odd");
        crypto_bigint::modular::MontyParams::new(modulus)
    }

    /// `i64` → coefficient field, with the test config.
    fn cf(v: i64) -> F {
        <F as FromWithConfig<i64>>::from_with_cfg(v, &test_cfg())
    }

    /// `PolyExt16` from 16 `i64` coefficients.
    fn poly(coeffs: [i64; 16]) -> P {
        P::from_coeffs(core::array::from_fn(|i| cf(coeffs[i])))
    }

    /// X^k as a `PolyExt16` (1 ≤ degree-monomial).
    fn monomial(k: usize) -> P {
        let mut c = [0i64; 16];
        c[k] = 1;
        poly(c)
    }

    #[test]
    fn zero_one_basics() {
        let cfg = test_cfg();
        let z = P::zero_with(&cfg);
        let o = P::one_with(&cfg);
        assert!(z.is_zero());
        assert!(!o.is_zero());
        assert!(P::is_zero(&z));
        assert_ne!(z, o);
        // additive / multiplicative identities
        let a = poly([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        assert_eq!(a.clone() + z.clone(), a);
        assert_eq!(a.clone() * o.clone(), a);
        assert_eq!(a.clone() * z.clone(), z);
    }

    #[test]
    fn add_is_commutative_and_associative() {
        let a = poly([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3]);
        let b = poly([2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9, 0, 4, 5]);
        let c = poly([1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 14, 23, 37, 60, 97]);
        // commutative
        assert_eq!(a.clone() + b.clone(), b.clone() + a.clone());
        // associative
        assert_eq!(
            (a.clone() + b.clone()) + c.clone(),
            a.clone() + (b.clone() + c.clone())
        );
    }

    #[test]
    fn sub_is_add_inverse() {
        let a = poly([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3]);
        let b = poly([2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9, 0, 4, 5]);
        assert_eq!((a.clone() - b.clone()) + b.clone(), a);
        assert_eq!(a.clone() - a.clone(), P::zero_with(&test_cfg()));
        assert_eq!(-(-a.clone()), a);
    }

    #[test]
    fn mul_is_commutative_and_associative() {
        let a = poly([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3]);
        let b = poly([2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9, 0, 4, 5]);
        let c = poly([1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0]);
        assert_eq!(a.clone() * b.clone(), b.clone() * a.clone());
        assert_eq!(
            (a.clone() * b.clone()) * c.clone(),
            a.clone() * (b.clone() * c.clone())
        );
    }

    #[test]
    fn mul_distributes_over_add() {
        let a = poly([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3]);
        let b = poly([2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9, 0, 4, 5]);
        let c = poly([1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 14, 23, 37, 60, 97]);
        let lhs = a.clone() * (b.clone() + c.clone());
        let rhs = a.clone() * b.clone() + a.clone() * c.clone();
        assert_eq!(lhs, rhs);
    }

    /// The defining `f̃`-fold: `X^16 ≡ X^5 + X^3 + X^2 + 1`.
    #[test]
    fn f_tilde_fold() {
        // X * X^15 = X^16, which must reduce to X^5 + X^3 + X^2 + 1.
        let x = monomial(1);
        let x15 = monomial(15);
        let x16 = x.clone() * x15.clone();

        let mut expected = [0i64; 16];
        for k in [0usize, 2, 3, 5] {
            expected[k] = 1;
        }
        assert_eq!(x16, poly(expected));

        // X^8 * X^8 = X^16 too — same reduction.
        let x8 = monomial(8);
        assert_eq!(x8.clone() * x8.clone(), poly(expected));

        // X^17 = X * X^16 = X*(X^5+X^3+X^2+1) = X^6 + X^4 + X^3 + X.
        let x17 = x.clone() * x16.clone();
        let mut expected17 = [0i64; 16];
        for k in [1usize, 3, 4, 6] {
            expected17[k] = 1;
        }
        assert_eq!(x17, poly(expected17));
    }

    /// `(X^15)^2 = X^30` must fold all the way down through f̃.
    #[test]
    fn high_degree_fold_nonzero() {
        let x15 = monomial(15);
        let x30 = x15.clone() * x15.clone();
        // X^30 reduces to a non-zero, fully-reduced (degree < 16) element.
        assert!(!x30.is_zero());
        // Cross-check: X^30 = X^14 * X^16 = X^14 * (X^5+X^3+X^2+1).
        let x14 = monomial(14);
        let x16 = monomial(8) * monomial(8);
        assert_eq!(x30, x14 * x16);
    }

    #[test]
    fn pow_matches_repeated_mul() {
        let a = poly([1, 2, 0, 3, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
        let mut acc = P::one_with(&test_cfg());
        for _ in 0..5 {
            acc = acc * a.clone();
        }
        assert_eq!(a.clone().pow(5), acc);
        assert_eq!(a.clone().pow(0), P::one_with(&test_cfg()));
        assert_eq!(a.clone().pow(1), a);
    }

    /// `FromWithConfig<&PackedI64Poly16>` round-trips a {0,1}-coef poly.
    #[test]
    fn from_packed_i64_round_trip() {
        let cfg = test_cfg();
        // A {0,1}-coefficient polynomial.
        let bits = [1i64, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1];
        let packed = PackedI64Poly16::<24>(DensePolynomial { coeffs: bits });
        let got = <P as FromWithConfig<&PackedI64Poly16<24>>>::from_with_cfg(&packed, &cfg);

        // Each {0,1} coefficient must land on MontyField 0 / 1.
        let expected = poly(bits);
        assert_eq!(got, expected);

        // Also check a signed (negative) coefficient reduces correctly:
        // -1 mod p must equal MontyField(0) - MontyField(1).
        let signed = [-1i64, 5, -7, 0, 100, -100, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let packed_s = PackedI64Poly16::<24>(DensePolynomial { coeffs: signed });
        let got_s = <P as FromWithConfig<&PackedI64Poly16<24>>>::from_with_cfg(&packed_s, &cfg);
        let expected_s = poly(signed);
        assert_eq!(got_s, expected_s);
        // -1 coefficient: confirm it equals 0 - 1 in the coefficient field.
        let neg_one = cf(0) - cf(1);
        assert_eq!(got_s.coeffs()[0], neg_one);
    }

    /// `FromWithConfig<&DensePolynomial<i64, 16>>` agrees with the
    /// packed conversion.
    #[test]
    fn from_dense_i64() {
        let cfg = test_cfg();
        let coeffs = [9i64, -3, 7, 0, 1, -1, 42, 0, 0, 0, 5, 0, 0, 0, 0, 8];
        let dp = DensePolynomial { coeffs };
        let got = <P as FromWithConfig<&DensePolynomial<i64, 16>>>::from_with_cfg(&dp, &cfg);
        assert_eq!(got, poly(coeffs));
    }

    /// `FromWithConfig<&BinaryPoly<16>>`: each bit → 0 / 1.
    #[test]
    fn from_binary_poly() {
        let cfg = test_cfg();
        let bp = BinaryPoly::<16>::from(0b1010_1100_0011_0101u64);
        let got = <P as FromWithConfig<&BinaryPoly<16>>>::from_with_cfg(&bp, &cfg);
        let mut expected = [0i64; 16];
        for (i, e) in expected.iter_mut().enumerate() {
            *e = ((0b1010_1100_0011_0101u64 >> i) & 1) as i64;
        }
        assert_eq!(got, poly(expected));
    }

    /// Transcript round-trip: `Inner` encode → decode is the identity.
    #[test]
    fn transcript_round_trip() {
        let cfg = test_cfg();
        let a = poly([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3]);
        let inner = a.clone().into_inner();

        let mut buf = vec![0u8; <PolyExt16Inner<LIMBS> as ConstTranscribable>::NUM_BYTES];
        inner.write_transcription_bytes_exact(&mut buf);
        let inner2 =
            <PolyExt16Inner<LIMBS> as GenTranscribable>::read_transcription_bytes_exact(&buf);
        assert_eq!(inner, inner2, "inner transcript round-trip mismatch");

        // Full field round-trip: into_inner -> new_unchecked_with_cfg.
        let a2 = <P as PrimeField>::new_unchecked_with_cfg(inner2, &cfg);
        assert_eq!(a, a2, "field element transcript round-trip mismatch");
    }

    /// `NUM_BYTES` is exactly 16 × one `MontyField::Inner` width.
    #[test]
    fn inner_num_bytes() {
        assert_eq!(
            <PolyExt16Inner<LIMBS> as ConstTranscribable>::NUM_BYTES,
            16 * <Uint<LIMBS> as ConstTranscribable>::NUM_BYTES
        );
        // With LIMBS = 4 (256-bit) that is 16 * 32 = 512 bytes.
        assert_eq!(
            <PolyExt16Inner<LIMBS> as ConstTranscribable>::NUM_BYTES,
            512
        );
    }

    /// PCS-path transcript round trip: mirror exactly what
    /// `pcs_transcript.rs` does — take a `PolyExt16`, borrow its
    /// `Field::inner()`, `write_transcription_bytes_exact` into a
    /// `NUM_BYTES`-sized buffer, `read_transcription_bytes_exact` it
    /// back, then `PrimeField::new_unchecked_with_cfg` to reconstruct,
    /// and assert equality. This is the path unblocked by making
    /// `inner()` a real borrow.
    ///
    /// Note `new_unchecked_with_cfg` (not `new_with_cfg`) is the correct
    /// reconstruction here, and is exactly what
    /// `pcs_transcript::read_field_element_no_length` uses: `inner()`
    /// yields the *Montgomery-form* `Uint`s, and `new_unchecked_with_cfg`
    /// stores them verbatim. `new_with_cfg` would treat them as ordinary
    /// integers and re-reduce, double-converting them.
    #[test]
    fn pcs_transcript_inner_round_trip() {
        let cfg = test_cfg();
        let a = poly([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3]);

        // `Field::inner()` must be a real, panic-free borrow.
        let inner_ref: &PolyExt16Inner<LIMBS> = <P as Field>::inner(&a);

        // Serialize exactly as `pcs_transcript::write_field_element_no_length`.
        let mut buf = vec![0u8; <PolyExt16Inner<LIMBS> as ConstTranscribable>::NUM_BYTES];
        inner_ref.write_transcription_bytes_exact(&mut buf);

        // Deserialize and reconstruct the field element, exactly as
        // `pcs_transcript::read_field_element_no_length`.
        let inner_back =
            <PolyExt16Inner<LIMBS> as GenTranscribable>::read_transcription_bytes_exact(&buf);
        let reconstructed = <P as PrimeField>::new_unchecked_with_cfg(inner_back, &cfg);
        assert_eq!(a, reconstructed, "PCS transcript round-trip mismatch");

        // `inner_mut()` is likewise a real borrow.
        let mut b = a.clone();
        let _inner_mut: &mut PolyExt16Inner<LIMBS> = <P as Field>::inner_mut(&mut b);
        assert_eq!(a, b);
    }

    /// `MulByScalar` agrees with `Mul`.
    #[test]
    fn mul_by_scalar_matches_mul() {
        let a = poly([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3]);
        let b = poly([2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9, 0, 4, 5]);
        let via_scalar = a.mul_by_scalar::<false>(&b).unwrap();
        assert_eq!(via_scalar, a.clone() * b.clone());
    }

    /// `FromWithConfig<&i128>` builds the degree-0 constant polynomial:
    /// the scalar (reduced mod p) in coefficient 0, rest zero.
    #[test]
    fn from_i128_constant_poly() {
        let cfg = test_cfg();
        for &v in &[0i128, 1, 2, 42, -1, -7, 1_000_000_007, -123_456_789] {
            let got = <P as FromWithConfig<&i128>>::from_with_cfg(&v, &cfg);
            let mut expected = [0i64; 16];
            // `i128` test values all fit in `i64` here.
            expected[0] = v as i64;
            assert_eq!(got, poly(expected), "i128 const-poly mismatch for {v}");
            // Coefficients 1..16 must be zero.
            for i in 1..16 {
                assert!(Coef::<LIMBS>::is_zero(&got.coeffs()[i]));
            }
        }
    }

    /// `FromWithConfig<&MontyField>` builds the constant polynomial whose
    /// coefficient 0 is the scalar (rest zero).
    #[test]
    fn from_monty_field_constant_poly() {
        let cfg = test_cfg();
        for &v in &[0i64, 1, 99, -5, 7_777_777] {
            let scalar = cf(v);
            let got = <P as FromWithConfig<&F>>::from_with_cfg(&scalar, &cfg);
            assert_eq!(got.coeffs()[0], scalar);
            for i in 1..16 {
                assert!(Coef::<LIMBS>::is_zero(&got.coeffs()[i]));
            }
        }
    }

    /// `FromPrimitiveWithConfig` blanket impl is satisfied: each
    /// primitive width yields the degree-0 constant polynomial.
    #[test]
    fn from_primitive_with_config() {
        use crypto_primitives::FromPrimitiveWithConfig;
        fn assert_fp<T: FromPrimitiveWithConfig>() {}
        assert_fp::<P>();

        let cfg = test_cfg();
        let a = <P as FromWithConfig<u8>>::from_with_cfg(200u8, &cfg);
        assert_eq!(a, poly([200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]));
        let b = <P as FromWithConfig<u64>>::from_with_cfg(1_234_567_890u64, &cfg);
        assert_eq!(
            b,
            poly([1_234_567_890, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        );
        let c = <P as FromWithConfig<i32>>::from_with_cfg(-99i32, &cfg);
        assert_eq!(c, poly([-99, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]));
        // Negative reduces correctly: -1 → 0 - 1 in the coefficient field.
        let neg = <P as FromWithConfig<i8>>::from_with_cfg(-1i8, &cfg);
        assert_eq!(neg.coeffs()[0], cf(0) - cf(1));
    }
}
