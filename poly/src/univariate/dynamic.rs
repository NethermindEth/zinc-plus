use crypto_primitives::{
    FieldConfig, ProjectElementWithConfig, RingConfig, SemiringConfig, SetConfig, boolean::Boolean,
};
use derive_more::From;
use itertools::Itertools;
use std::fmt::Display;
use zinc_transcript::traits::{ConstTranscribable, GenTranscribable, Transcribable};
use zinc_utils::{add, mul, projectable_to_field::ProjectableToField, rem};

use crate::{
    EvaluationError,
    univariate::{binary::BinaryPoly, dense::DensePolynomial},
};

#[allow(clippy::arithmetic_side_effects)]
fn new_coeffs_trimmed<R: Clone>(coeffs: &[R], is_zero: impl Fn(&R) -> bool) -> Vec<R> {
    if let Some((non_zero, _)) = coeffs.iter().rev().find_position(|&coeff| !is_zero(coeff)) {
        let deg_plus_one = coeffs.len() - non_zero;

        coeffs.iter().take(deg_plus_one).cloned().collect()
    } else {
        Vec::new()
    }
}

#[allow(clippy::arithmetic_side_effects)]
fn degree<R>(coeffs: &[R], is_zero: impl Fn(&R) -> bool) -> Option<usize> {
    coeffs
        .iter()
        .rev()
        .find_position(|coeff| !is_zero(coeff))
        .map(|(non_zero, _)| coeffs.len() - non_zero - 1)
}

#[allow(clippy::arithmetic_side_effects)]
fn trim<R>(coeffs: &mut Vec<R>, is_zero: impl Fn(&R) -> bool) {
    coeffs.truncate(degree(coeffs, is_zero).map_or(0, |degree| degree + 1))
}

fn is_zero<R>(coeffs: &[R], is_zero: impl Fn(&R) -> bool) -> bool {
    coeffs.iter().all(is_zero)
}

/// Polynomials of dynamic degree over an arbitrary semiring (fixed like
/// `Int`, or dynamic like a random finite field). To be used in UAIR and
/// PIOP where ZIP+ degree bound is not observed anymore.
///
/// This is a dumb data holder: all operations are performed via
/// [`DynamicPolynomialConfig`], obtainable from the coefficient config with
/// [`HasDynamicPolynomialConfig::dyn_poly_cfg`].
///
/// Note that operations involving dynamic polynomials
/// do not trim leading zeros meaning
/// one can end up with unequal objects of the type
/// `DynamicPolynomial<E>` that represent equal polynomials,
/// therefore [`DynamicPolynomialConfig::trim`] has to be called before
/// checking equality.
#[derive(Debug, Clone, From, Hash, PartialEq, Eq)]
pub struct DynamicPolynomial<E> {
    pub coeffs: Vec<E>,
}

impl<E> DynamicPolynomial<E> {
    pub const ZERO: Self = Self { coeffs: Vec::new() };

    /// Maps every field element through `f`, preserving structure — used to
    /// lift elements into wire integers and to project wire integers back
    /// into elements at the (de)serialization boundary.
    pub fn try_map<T, Er>(
        &self,
        f: impl FnMut(&E) -> Result<T, Er> + Copy,
    ) -> Result<DynamicPolynomial<T>, Er> {
        Ok(DynamicPolynomial {
            coeffs: self.coeffs.iter().map(f).try_collect()?,
        })
    }

    /// Create a new polynomial with the given coefficients.
    #[inline(always)]
    pub fn new(coeffs: impl AsRef<[E]>) -> Self
    where
        E: Clone,
    {
        Self {
            coeffs: Vec::from(coeffs.as_ref()),
        }
    }
}

impl<E> Default for DynamicPolynomial<E> {
    fn default() -> Self {
        Self::ZERO
    }
}

impl<E> FromIterator<E> for DynamicPolynomial<E> {
    #[inline(always)]
    fn from_iter<T: IntoIterator<Item = E>>(iter: T) -> Self {
        Self {
            coeffs: iter.into_iter().collect(),
        }
    }
}

impl<E: Display> Display for DynamicPolynomial<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        let mut first = true;

        for coeff in self.coeffs.iter() {
            if first {
                first = false;
            } else {
                write!(f, ", ")?;
            }
            write!(f, "{}", coeff)?;
        }

        write!(f, "]")
    }
}

impl<E, const DEGREE_PLUS_ONE: usize> From<DensePolynomial<E, DEGREE_PLUS_ONE>>
    for DynamicPolynomial<E>
{
    fn from(dense_poly: DensePolynomial<E, DEGREE_PLUS_ONE>) -> Self {
        Self {
            coeffs: Vec::from(dense_poly.coeffs),
        }
    }
}

impl<const DEGREE_PLUS_ONE: usize> From<BinaryPoly<DEGREE_PLUS_ONE>>
    for DynamicPolynomial<Boolean>
{
    fn from(binary_poly: BinaryPoly<DEGREE_PLUS_ONE>) -> Self {
        Self::from(DensePolynomial::from(binary_poly))
    }
}

/// Configuration of the polynomial semiring $S[X]$ over the (semi)ring
/// configured by `S`, with [`DynamicPolynomial`] as its element.
///
/// Implements exactly the layer its coefficients provide: [`SemiringConfig`]
/// over a semiring, additionally [`RingConfig`] over a ring. The polynomial
/// ring is never a field, hence no [`FieldConfig`].
///
/// Checked operations delegate to the coefficient config's checked
/// operations, so overflow behavior follows the coefficients (e.g. `Int`
/// coefficients can overflow, field coefficients cannot).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DynamicPolynomialConfig<'a, S: SemiringConfig> {
    pub cfg: &'a S,
}

/// Extension trait providing [`DynamicPolynomialConfig`] from a coefficient
/// config, so that the same config can be used to perform operations on
/// dynamic polynomials: `cfg.poly_cfg().mul(&p, &q)`.
pub trait HasDynamicPolynomialConfig: SemiringConfig + Sized {
    #[inline(always)]
    fn dyn_poly_cfg(&self) -> DynamicPolynomialConfig<'_, Self> {
        DynamicPolynomialConfig { cfg: self }
    }
}

impl<S: SemiringConfig> HasDynamicPolynomialConfig for S {}

impl<'a, S: SemiringConfig> SetConfig for DynamicPolynomialConfig<'a, S> {
    type Element = DynamicPolynomial<S::Element>;
}

impl<'a, S: SemiringConfig> SemiringConfig for DynamicPolynomialConfig<'a, S> {
    fn is_zero(&self, value: &Self::Element) -> bool {
        is_zero(&value.coeffs, |e| self.cfg.is_zero(e))
    }

    fn zero(&self) -> Self::Element {
        DynamicPolynomial::ZERO
    }

    fn one(&self) -> Self::Element {
        DynamicPolynomial {
            coeffs: vec![self.cfg.one()],
        }
    }

    fn add(&self, x: &Self::Element, y: &Self::Element) -> Self::Element {
        let mut res = x.clone();
        self.add_assign(&mut res, y);
        res
    }

    fn sub(&self, x: &Self::Element, y: &Self::Element) -> Self::Element {
        let mut res = x.clone();
        self.sub_assign(&mut res, y);
        res
    }

    fn mul(&self, x: &Self::Element, y: &Self::Element) -> Self::Element {
        if self.is_zero(x) || self.is_zero(y) {
            return self.zero();
        }
        let mut coeffs =
            vec![self.cfg.zero(); add!(x.coeffs.len(), y.coeffs.len()).saturating_sub(1)];
        for (i, a) in x.coeffs.iter().enumerate() {
            for (j, b) in y.coeffs.iter().enumerate() {
                let prod = self.cfg.mul(a, b);
                self.cfg.add_assign(&mut coeffs[add!(i, j)], &prod);
            }
        }
        DynamicPolynomial { coeffs }
    }

    fn pow_u32(&self, x: &Self::Element, y: u32) -> Self::Element {
        let mut res = self.one();
        for _ in 0..y {
            res = self.mul(&res, x);
        }
        res
    }

    fn checked_add(&self, x: &Self::Element, y: &Self::Element) -> Option<Self::Element> {
        let mut res = x.clone();
        if res.coeffs.len() < y.coeffs.len() {
            res.coeffs.resize(y.coeffs.len(), self.cfg.zero());
        }
        for (xc, yc) in res.coeffs.iter_mut().zip(&y.coeffs) {
            *xc = self.cfg.checked_add(xc, yc)?;
        }
        Some(res)
    }

    fn checked_sub(&self, x: &Self::Element, y: &Self::Element) -> Option<Self::Element> {
        let mut res = x.clone();
        if res.coeffs.len() < y.coeffs.len() {
            res.coeffs.resize(y.coeffs.len(), self.cfg.zero());
        }
        for (xc, yc) in res.coeffs.iter_mut().zip(&y.coeffs) {
            *xc = self.cfg.checked_sub(xc, yc)?;
        }
        Some(res)
    }

    fn checked_mul(&self, x: &Self::Element, y: &Self::Element) -> Option<Self::Element> {
        if self.is_zero(x) || self.is_zero(y) {
            return Some(self.zero());
        }
        let mut coeffs =
            vec![self.cfg.zero(); add!(x.coeffs.len(), y.coeffs.len()).saturating_sub(1)];
        for (i, a) in x.coeffs.iter().enumerate() {
            for (j, b) in y.coeffs.iter().enumerate() {
                let prod = self.cfg.checked_mul(a, b)?;
                let idx = add!(i, j);
                coeffs[idx] = self.cfg.checked_add(&coeffs[idx], &prod)?;
            }
        }
        Some(DynamicPolynomial { coeffs })
    }

    fn checked_pow_u32(&self, x: &Self::Element, y: u32) -> Option<Self::Element> {
        let mut res = self.one();
        for _ in 0..y {
            res = self.checked_mul(&res, x)?;
        }
        Some(res)
    }

    fn add_assign(&self, x: &mut Self::Element, y: &Self::Element) {
        if x.coeffs.len() < y.coeffs.len() {
            x.coeffs.resize(y.coeffs.len(), self.cfg.zero());
        }
        for (xc, yc) in x.coeffs.iter_mut().zip(&y.coeffs) {
            self.cfg.add_assign(xc, yc);
        }
    }

    fn sub_assign(&self, x: &mut Self::Element, y: &Self::Element) {
        if x.coeffs.len() < y.coeffs.len() {
            x.coeffs.resize(y.coeffs.len(), self.cfg.zero());
        }
        for (xc, yc) in x.coeffs.iter_mut().zip(&y.coeffs) {
            self.cfg.sub_assign(xc, yc);
        }
    }
}

impl<'a, S: RingConfig> RingConfig for DynamicPolynomialConfig<'a, S> {
    fn neg(&self, x: &Self::Element) -> Self::Element {
        DynamicPolynomial {
            coeffs: x.coeffs.iter().map(|c| self.cfg.neg(c)).collect(),
        }
    }

    fn checked_neg(&self, x: &Self::Element) -> Option<Self::Element> {
        Some(DynamicPolynomial {
            coeffs: x
                .coeffs
                .iter()
                .map(|c| self.cfg.checked_neg(c))
                .collect::<Option<Vec<_>>>()?,
        })
    }
}

impl<'a, S: SemiringConfig> DynamicPolynomialConfig<'a, S> {
    /// Create a new polynomial with the given coefficients, trimming the
    /// leading zeros.
    #[inline(always)]
    pub fn new_trimmed(&self, coeffs: impl AsRef<[S::Element]>) -> DynamicPolynomial<S::Element> {
        DynamicPolynomial {
            coeffs: new_coeffs_trimmed(coeffs.as_ref(), |e| self.cfg.is_zero(e)),
        }
    }

    #[inline(always)]
    pub fn degree(&self, poly: &DynamicPolynomial<S::Element>) -> Option<usize> {
        degree(&poly.coeffs, |e| self.cfg.is_zero(e))
    }

    #[inline(always)]
    pub fn trim(&self, poly: &mut DynamicPolynomial<S::Element>) {
        trim(&mut poly.coeffs, |e| self.cfg.is_zero(e));
    }

    /// Evaluate the polynomial at the given point using Horner's method.
    pub fn evaluate_at_point(
        &self,
        poly: &DynamicPolynomial<S::Element>,
        point: &S::Element,
    ) -> Result<S::Element, EvaluationError> {
        let mut result = poly.coeffs.last().cloned().unwrap_or(self.cfg.zero());

        for coeff in poly.coeffs.iter().rev().skip(1) {
            self.cfg.mul_assign(&mut result, point);
            self.cfg.add_assign(&mut result, coeff);
        }

        Ok(result)
    }

    /// Evaluate the polynomial at the given point using Horner's method,
    /// with overflow-checked coefficient operations.
    pub fn checked_evaluate_at_point(
        &self,
        poly: &DynamicPolynomial<S::Element>,
        point: &S::Element,
    ) -> Result<S::Element, EvaluationError> {
        let mut result = poly.coeffs.last().cloned().unwrap_or(self.cfg.zero());

        for coeff in poly.coeffs.iter().rev().skip(1) {
            let term = self
                .cfg
                .checked_mul(&result, point)
                .ok_or(EvaluationError::Overflow)?;
            result = self
                .cfg
                .checked_add(&term, coeff)
                .ok_or(EvaluationError::Overflow)?;
        }

        Ok(result)
    }

    /// Right-rotate the coefficient vector by `c` positions within width `D`.
    ///
    /// The output coefficient at position `i` is the input coefficient at
    /// `(i + c) mod D`. Missing coefficients are padded with zero before the
    /// rotation.
    pub fn rotate_right<const D: usize>(
        &self,
        poly: &DynamicPolynomial<S::Element>,
        c: usize,
    ) -> DynamicPolynomial<S::Element> {
        assert!(
            c > 0 && c < D,
            "rotate_right count {c} out of range (must satisfy 0 < c < {D})",
        );
        assert!(
            poly.coeffs.len() <= D,
            "rotate_right coefficient length {} exceeds width {D}",
            poly.coeffs.len(),
        );

        let mut coeffs = poly.coeffs.clone();
        coeffs.resize(D, self.cfg.zero());
        DynamicPolynomial {
            coeffs: (0..D)
                .map(|i| coeffs[rem!(add!(i, c), D)].clone())
                .collect(),
        }
    }

    /// Right-shift the coefficient vector by `c` positions within width `D`.
    ///
    /// The output coefficient at position `i` is the input coefficient at
    /// `i + c`, or zero when that index is outside width `D`. Missing
    /// coefficients are padded with zero before the shift.
    pub fn shr<const D: usize>(
        &self,
        poly: &DynamicPolynomial<S::Element>,
        c: usize,
    ) -> DynamicPolynomial<S::Element> {
        assert!(
            c > 0 && c < D,
            "shr count {c} out of range (must satisfy 0 < c < {D})",
        );
        assert!(
            poly.coeffs.len() <= D,
            "shr coefficient length {} exceeds width {D}",
            poly.coeffs.len(),
        );

        let zero = self.cfg.zero();
        let mut coeffs = poly.coeffs.clone();
        coeffs.resize(D, zero.clone());
        DynamicPolynomial {
            coeffs: (0..D)
                .map(|i| {
                    let j = add!(i, c);
                    if j < D {
                        coeffs[j].clone()
                    } else {
                        zero.clone()
                    }
                })
                .collect(),
        }
    }
}

/// Projection by evaluation: a dynamic polynomial over projectable
/// coefficients maps to the field element obtained by projecting the
/// coefficients and evaluating at the sampled point.
impl<E, C> ProjectableToField<C> for DynamicPolynomial<E>
where
    C: FieldConfig + ProjectElementWithConfig<E> + Clone + Send + Sync + 'static,
{
    fn prepare_projection(cfg: &C, sampled_value: &C::Element) -> impl Fn(&Self) -> C::Element {
        let cfg = cfg.clone();
        let sampled_value = sampled_value.clone();

        move |poly: &Self| {
            // Horner's method, projecting coefficients on the fly.
            let mut result = cfg.zero();
            for coeff in poly.coeffs.iter().rev() {
                cfg.mul_assign(&mut result, &sampled_value);
                let projected = cfg.project(coeff);
                cfg.add_assign(&mut result, &projected);
            }
            result
        }
    }
}

/// Unfortunately, we cannot implement of `GenTranscribable` for
/// `Vec<DynamicPolynomial<E>>` since they both are foreign types, so we define
/// this wrapper as a workaround.
///
/// Polynomials are transcribed as raw elements without any field metadata:
/// the field config is bound into the transcript separately, when the field
/// is sampled. Polynomials are written untrimmed (trimming requires a config),
/// so writers should trim beforehand if proof size matters.
#[derive(Debug, Default, Clone, From, Hash, PartialEq, Eq)]
#[repr(transparent)]
pub struct DynamicPolyVec<E>(pub Vec<DynamicPolynomial<E>>);

impl<E> DynamicPolyVec<E> {
    pub fn reinterpret(value: &Vec<DynamicPolynomial<E>>) -> &Self {
        // Safety: `DynamicPolyVec<E>` is a transparent wrapper, so the memory
        // layout is the same.
        unsafe { &*(value as *const Vec<DynamicPolynomial<E>> as *const Self) }
    }
}

impl<E> GenTranscribable for DynamicPolyVec<E>
where
    E: ConstTranscribable,
{
    fn read_transcription_bytes_exact(mut bytes: &[u8]) -> Self {
        let mut result = Vec::new();
        while !bytes.is_empty() {
            let (len, rest) = u32::read_transcription_bytes_subset(bytes);
            let len = usize::try_from(len).expect("polynomial length must fit into usize");
            bytes = rest;
            let end = mul!(len, E::NUM_BYTES);
            let coeffs: Vec<E> = Vec::read_transcription_bytes_exact(&bytes[..end]);
            result.push(DynamicPolynomial { coeffs });
            bytes = &bytes[end..];
        }
        result.into()
    }

    fn write_transcription_bytes_exact(&self, mut buf: &mut [u8]) {
        for poly in self.0.iter() {
            let len = u32::try_from(poly.coeffs.len()).expect("poly length must fit into u32");
            len.write_transcription_bytes_exact(&mut buf[0..u32::NUM_BYTES]);
            buf = &mut buf[u32::NUM_BYTES..];

            let end = mul!(poly.coeffs.len(), E::NUM_BYTES);
            poly.coeffs.write_transcription_bytes_exact(&mut buf[..end]);
            buf = &mut buf[end..];
        }
        assert!(buf.is_empty(), "Entire buffer should be used");
    }
}

impl<E> Transcribable for DynamicPolyVec<E>
where
    E: ConstTranscribable,
{
    fn get_num_bytes(&self) -> usize {
        self.0
            .iter()
            .map(|poly| add!(u32::NUM_BYTES, mul!(poly.coeffs.len(), E::NUM_BYTES)))
            .sum()
    }
}

#[cfg(test)]
#[allow(
    clippy::arithmetic_side_effects,
    clippy::clone_on_copy,
    clippy::redundant_clone
)]
mod field_tests {
    use crypto_primitives::{
        BaseFieldConfig, ProjectElementWithConfig,
        crypto_bigint_monty::{MontyField, MontyFieldElement},
        crypto_bigint_uint::Uint,
    };

    use super::*;

    const LIMBS: usize = 4;
    type F = MontyField<LIMBS>;
    type P = DynamicPolynomial<MontyFieldElement<LIMBS>>;

    fn field_config() -> F {
        let modulus =
            Uint::from_be_hex("0000000000000000000000000000000000860995AE68FC80E1B1BD1E39D54B33");
        F::new(&modulus).expect("modulus should be a valid odd prime")
    }

    fn f(v: i64) -> MontyFieldElement<LIMBS> {
        field_config().project(&v)
    }

    fn p(coeffs: impl IntoIterator<Item = i64>) -> P {
        coeffs.into_iter().map(f).collect()
    }

    #[test]
    fn new_trimmed_creates_correctly() {
        let field_cfg = field_config();
        let cfg = field_cfg.dyn_poly_cfg();
        assert_eq!(cfg.new_trimmed(p([1, 2, 3, 0, 0]).coeffs), p([1, 2, 3]));
    }

    #[test]
    fn add_zero() {
        let field_cfg = field_config();
        let cfg = field_cfg.dyn_poly_cfg();

        assert_eq!(cfg.add(&P::ZERO, &P::ZERO), P::ZERO);

        let x = p([2, 0, 2, 0, 0]);
        assert_eq!(cfg.add(&x, &P::ZERO), x);
        assert_eq!(cfg.add(&P::ZERO, &x), x);

        let mut y = x.clone();
        cfg.add_assign(&mut y, &P::ZERO);
        assert_eq!(y, x);
    }

    #[test]
    fn addition_is_correct() {
        let field_cfg = field_config();
        let cfg = field_cfg.dyn_poly_cfg();
        let (x, y) = (p([2, 0, 2, 0, 0]), p([1, 2, 3]));

        let res = p([3, 2, 5, 0, 0]);

        assert_eq!(cfg.add(&x, &y), res);
        assert_eq!(cfg.add(&y, &x), res);
        assert_eq!(cfg.checked_add(&x, &y), Some(res.clone()));

        let mut z = x.clone();
        cfg.add_assign(&mut z, &y);
        assert_eq!(z, res);
    }

    #[test]
    fn subtraction_is_correct() {
        let field_cfg = field_config();
        let cfg = field_cfg.dyn_poly_cfg();
        let (x, y) = (p([2, 0, 2, 0, 0]), p([1, 2, 3]));

        let res = p([1, -2, -1, 0, 0]);

        assert_eq!(cfg.sub(&x, &y), res);
        assert_eq!(cfg.checked_sub(&x, &y), Some(res.clone()));

        let mut z = x.clone();
        cfg.sub_assign(&mut z, &y);
        assert_eq!(z, res);

        // Subtraction with the result longer than the lhs
        assert_eq!(
            cfg.sub(&p([1, 2, 3]), &p([2, 0, 2, -1, 0])),
            p([-1, 2, 1, 1, 0])
        );
    }

    #[test]
    fn multiplication_is_correct() {
        let field_cfg = field_config();
        let cfg = field_cfg.dyn_poly_cfg();
        let (x, y) = (p([2, 0, 2]), p([1, 2, 3]));

        let res = p([2, 4, 8, 4, 6]);

        assert_eq!(cfg.mul(&x, &y), res);
        assert_eq!(cfg.mul(&y, &x), res);
        assert_eq!(cfg.checked_mul(&x, &y), Some(res));

        assert_eq!(cfg.mul(&x, &cfg.zero()), cfg.zero());
        assert_eq!(cfg.mul(&cfg.zero(), &x), cfg.zero());
    }

    #[test]
    fn test_trim() {
        let field_cfg = field_config();
        let cfg = field_cfg.dyn_poly_cfg();

        let mut x = p([0, 0, 0, 0, 0]);
        cfg.trim(&mut x);
        assert_eq!(x, P::ZERO);

        let mut x = p([2, 3, 0, 0, 0]);
        cfg.trim(&mut x);
        assert_eq!(x, p([2, 3]));
    }

    #[test]
    fn evaluate_zero_poly() {
        let field_cfg = field_config();
        let cfg = field_cfg.dyn_poly_cfg();
        assert_eq!(cfg.evaluate_at_point(&P::ZERO, &f(1)), Ok(f(0)))
    }

    #[test]
    fn evaluation_is_correct() {
        let field_cfg = field_config();
        let cfg = field_cfg.dyn_poly_cfg();
        // 1 + 2x + 3x² at x = 2 → 1 + 4 + 12 = 17
        assert_eq!(cfg.evaluate_at_point(&p([1, 2, 3]), &f(2)), Ok(f(17)));
    }

    #[test]
    fn projection_evaluates_at_sampled_point() {
        use crypto_primitives::crypto_bigint_int::Int;
        type IntPoly = DynamicPolynomial<Int<4>>;

        let field_cfg = field_config();
        let sampled = f(2);
        let project = IntPoly::prepare_projection(&field_cfg, &sampled);
        // 1 + 2x + 3x² at x = 2 → 17
        let poly: IntPoly = [1i8, 2, 3].map(Int::from_i8).into_iter().collect();
        assert_eq!(project(&poly), f(17));
        assert_eq!(project(&IntPoly::ZERO), f(0));
    }

    #[test]
    fn rotate_right_pads_and_permutes_coefficients() {
        let field_cfg = field_config();
        let cfg = field_cfg.dyn_poly_cfg();
        assert_eq!(cfg.rotate_right::<5>(&p([1, 2, 3]), 2), p([3, 0, 0, 1, 2]));
    }

    #[test]
    fn shr_pads_and_drops_coefficients() {
        let field_cfg = field_config();
        let cfg = field_cfg.dyn_poly_cfg();
        assert_eq!(cfg.shr::<5>(&p([1, 2, 3]), 2), p([3, 0, 0, 0, 0]));
    }

    #[test]
    #[should_panic(expected = "rotate_right count 0 out of range")]
    fn rotate_right_panics_on_zero() {
        let field_cfg = field_config();
        let _ = field_cfg.dyn_poly_cfg().rotate_right::<5>(&p([1]), 0);
    }

    #[test]
    #[should_panic(expected = "shr count 5 out of range")]
    fn shr_panics_on_full_width() {
        let field_cfg = field_config();
        let _ = field_cfg.dyn_poly_cfg().shr::<5>(&p([1]), 5);
    }
}

#[cfg(test)]
#[allow(clippy::clone_on_copy, clippy::redundant_clone)]
mod semiring_tests {
    use crypto_primitives::{FixedConfig, crypto_bigint_int::Int};
    use num_traits::ConstZero;

    use super::*;

    type R = Int<4>;
    type P = DynamicPolynomial<R>;
    type FC = FixedConfig<R>;

    fn p(coeffs: impl IntoIterator<Item = i8>) -> P {
        coeffs.into_iter().map(Int::from_i8).collect()
    }

    fn get_2_test_polynomials() -> (P, P) {
        (p([2, 0, 2, 0, 0]), p([1, 2, 3]))
    }

    #[test]
    fn new_trimmed_creates_correctly() {
        let int_cfg = FC::default();
        let cfg = int_cfg.dyn_poly_cfg();
        assert_eq!(cfg.new_trimmed(p([1, 2, 3, 0, 0]).coeffs), p([1, 2, 3]));
    }

    #[test]
    fn add_zero() {
        let int_cfg = FC::default();
        let cfg = int_cfg.dyn_poly_cfg();

        assert_eq!(cfg.add(&P::ZERO, &P::ZERO), P::ZERO);

        let x = p([2, 0, 2, 0, 0]);
        assert_eq!(cfg.add(&x, &P::ZERO), x);
        assert_eq!(cfg.add(&P::ZERO, &x), x);

        let mut y = x.clone();
        cfg.add_assign(&mut y, &P::ZERO);
        assert_eq!(y, x);
    }

    #[test]
    fn addition_is_correct() {
        let int_cfg = FC::default();
        let cfg = int_cfg.dyn_poly_cfg();
        let (x, y) = get_2_test_polynomials();

        let res = p([3, 2, 5, 0, 0]);

        assert_eq!(cfg.add(&x, &y), res);
        assert_eq!(cfg.add(&y, &x), res);
        assert_eq!(cfg.checked_add(&x, &y), Some(res.clone()));
        assert_eq!(cfg.checked_add(&y, &x), Some(res.clone()));

        let mut z = x.clone();
        cfg.add_assign(&mut z, &y);
        assert_eq!(z, res);

        let mut z = y.clone();
        cfg.add_assign(&mut z, &x);
        assert_eq!(z, res);
    }

    #[test]
    fn subtraction_is_correct() {
        let int_cfg = FC::default();
        let cfg = int_cfg.dyn_poly_cfg();
        let (x, y) = get_2_test_polynomials();

        let res = p([1, -2, -1, 0, 0]);

        assert_eq!(cfg.sub(&x, &y), res);
        assert_eq!(cfg.checked_sub(&x, &y), Some(res.clone()));

        let mut z = x.clone();
        cfg.sub_assign(&mut z, &y);
        assert_eq!(z, res);

        // Subtraction with the result longer than the lhs
        assert_eq!(
            cfg.sub(&p([1, 2, 3]), &p([2, 0, 2, -1, 0])),
            p([-1, 2, 1, 1, 0])
        );
    }

    #[test]
    fn mul_zero() {
        let int_cfg = FC::default();
        let cfg = int_cfg.dyn_poly_cfg();

        assert_eq!(cfg.mul(&P::ZERO, &P::ZERO), P::ZERO);

        let x = p([2, 0, 2, 0, 0]);
        assert_eq!(cfg.mul(&x, &P::ZERO), P::ZERO);
        assert_eq!(cfg.mul(&P::ZERO, &x), P::ZERO);
        assert_eq!(cfg.checked_mul(&x, &P::ZERO), Some(P::ZERO));
    }

    #[test]
    fn multiplication_is_correct() {
        let int_cfg = FC::default();
        let cfg = int_cfg.dyn_poly_cfg();
        let (x, y) = get_2_test_polynomials();

        // Untrimmed operands: result carries the trailing zeros.
        let res = p([2, 4, 8, 4, 6, 0, 0]);

        assert_eq!(cfg.mul(&x, &y), res);
        assert_eq!(cfg.mul(&y, &x), res);
        assert_eq!(cfg.checked_mul(&x, &y), Some(res.clone()));
        assert_eq!(cfg.checked_mul(&y, &x), Some(res));
    }

    #[test]
    fn negation_is_correct() {
        let int_cfg = FC::default();
        let cfg = int_cfg.dyn_poly_cfg();
        let x = p([1, -2, 3]);

        assert_eq!(cfg.neg(&x), p([-1, 2, -3]));
        assert_eq!(cfg.checked_neg(&x), Some(p([-1, 2, -3])));
    }

    #[test]
    fn checked_ops_detect_coefficient_overflow() {
        let int_cfg = FC::default();
        let cfg = int_cfg.dyn_poly_cfg();

        let max = P::new([Int::MAX]);
        let one = cfg.one();

        assert_eq!(cfg.checked_add(&max, &one), None);
        assert_eq!(cfg.checked_mul(&max, &p([2])), None);

        let min = P::new([Int::MIN]);
        assert_eq!(cfg.checked_sub(&min, &one), None);
        assert_eq!(cfg.checked_neg(&min), None);
    }

    #[test]
    fn checked_evaluation_detects_overflow() {
        let int_cfg = FC::default();
        let cfg = int_cfg.dyn_poly_cfg();

        // 1 + 2x + 3x² at x = 2 → 17
        assert_eq!(
            cfg.checked_evaluate_at_point(&p([1, 2, 3]), &Int::from_i8(2)),
            Ok(Int::from_i8(17))
        );
        assert_eq!(
            cfg.checked_evaluate_at_point(&P::new([Int::ZERO, Int::MAX]), &Int::from_i8(2)),
            Err(EvaluationError::Overflow)
        );
    }

    #[test]
    fn test_trim() {
        let int_cfg = FC::default();
        let cfg = int_cfg.dyn_poly_cfg();

        let mut x = p([0, 0, 0, 0, 0]);
        cfg.trim(&mut x);
        assert_eq!(x, P::ZERO);

        let mut x = p([2, 3, 0, 0, 0]);
        cfg.trim(&mut x);
        assert_eq!(x, p([2, 3]));
    }
}
