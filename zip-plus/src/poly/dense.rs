use super::{ConstCoeffBitWidth, EvaluatablePolynomial, EvaluationError};
use crate::{
    pcs::structs::{MulByScalar, ProjectableToField},
    traits::{ConstTranscribable, FromRef, Named},
};
use crypto_primitives::{FromWithConfig, IntoWithConfig, PrimeField, Ring, Semiring};
use itertools::Itertools;
use num_traits::{CheckedAdd, CheckedMul, CheckedNeg, CheckedSub, One, Zero};
use rand::{distr::StandardUniform, prelude::*};
use std::{
    array,
    fmt::Display,
    hash::Hash,
    iter,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};
// TODO: rename to univariate?

// Sadly, we cannot use [R; DEGREE + 1] in stable Rust yet, so we use separate
// coeff_0.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DensePolynomial<R, const DEGREE: usize> {
    /// Coefficient of the polynomial of degree 0.
    coeff_0: R,

    /// Coefficients of the polynomial, lowest degree first, starting with
    /// degree 1.
    coeffs: [R; DEGREE],
}

impl<R: Semiring + Zero, const DEGREE: usize> DensePolynomial<R, DEGREE> {
    /// Create a new polynomial with the given coefficients.
    /// If the input has fewer than N+1 coefficients, the remaining slots will
    /// be filled with zeros. If the input has more than N+1 coefficients,
    /// it will panic.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn new(coeffs: impl AsRef<[R]>) -> Self {
        let coeffs = coeffs.as_ref();
        assert!(
            coeffs.len() <= DEGREE + 1,
            "Too many coefficients provided: expected at most {}, got {}",
            DEGREE + 1,
            coeffs.len()
        );

        if coeffs.is_empty() {
            return Self::zero();
        }

        let coeff_0 = coeffs[0].clone();
        let mut coeffs = coeffs[1..].to_vec();
        coeffs.resize(DEGREE, R::zero());
        let coeffs = coeffs.try_into().expect("unreachable");

        DensePolynomial { coeff_0, coeffs }
    }

    /// Return all coefficients as a vector.
    /// The result contains DEGREE+1 elements: [coeff_0, coeffs[0], ...,
    /// coeffs[DEGREE-1]].
    #[allow(clippy::arithmetic_side_effects)]
    pub fn to_coeffs(&self) -> Vec<R> {
        let mut result = Vec::with_capacity(DEGREE + 1);
        result.push(self.coeff_0.clone());
        result.extend_from_slice(&self.coeffs);
        result
    }
}

impl<R: Copy, const DEGREE: usize> Copy for DensePolynomial<R, DEGREE> {}

impl<R: Default, const DEGREE: usize> Default for DensePolynomial<R, DEGREE> {
    fn default() -> Self {
        DensePolynomial {
            coeff_0: R::default(),
            coeffs: array::from_fn::<_, DEGREE, _>(|_| R::default()),
        }
    }
}

impl<R: Display, const DEGREE: usize> Display for DensePolynomial<R, DEGREE> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        write!(f, "{}", self.coeff_0)?;
        for coeff in self.coeffs.iter() {
            write!(f, ", ")?;
            write!(f, "{}", coeff)?;
        }
        write!(f, "]")?;
        Ok(())
    }
}

impl<R: Hash, const DEGREE: usize> Hash for DensePolynomial<R, DEGREE> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.coeff_0.hash(state);
        for coeff in self.coeffs.iter() {
            coeff.hash(state);
        }
    }
}

impl<R: Semiring + Zero, const DEGREE: usize> Zero for DensePolynomial<R, DEGREE> {
    fn zero() -> Self {
        Self {
            coeff_0: R::zero(),
            coeffs: array::from_fn::<_, DEGREE, _>(|_| R::zero()),
        }
    }

    fn is_zero(&self) -> bool {
        self.coeff_0.is_zero() && self.coeffs.iter().all(|c| c.is_zero())
    }
}

impl<R: Semiring + Zero + One, const DEGREE: usize> One for DensePolynomial<R, DEGREE> {
    fn one() -> Self {
        Self {
            coeff_0: R::one(),
            coeffs: array::from_fn::<_, DEGREE, _>(|_| R::zero()),
        }
    }
}

impl<R: Ring + Neg<Output = R>, const DEGREE: usize> Neg for DensePolynomial<R, DEGREE> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)] // By design
    fn neg(mut self) -> Self::Output {
        self.coeff_0 = -self.coeff_0;
        self.coeffs.iter_mut().for_each(|c| *c = -c.clone());
        self
    }
}

impl<R: Semiring, const DEGREE: usize> Add for DensePolynomial<R, DEGREE> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects, clippy::op_ref)]
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        self + &rhs
    }
}

impl<'a, R: Semiring, const DEGREE: usize> Add<&'a Self> for DensePolynomial<R, DEGREE> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn add(mut self, rhs: &'a Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<R: Semiring, const DEGREE: usize> Sub for DensePolynomial<R, DEGREE> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects, clippy::op_ref)]
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        self - &rhs
    }
}

impl<'a, R: Semiring, const DEGREE: usize> Sub<&'a Self> for DensePolynomial<R, DEGREE> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn sub(mut self, rhs: &'a Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl<R: Semiring, const DEGREE: usize> Mul for DensePolynomial<R, DEGREE> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects, clippy::op_ref)]
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        self * &rhs
    }
}

impl<'a, R: Semiring, const DEGREE: usize> Mul<&'a Self> for DensePolynomial<R, DEGREE> {
    type Output = Self;

    fn mul(self, _rhs: &'a Self) -> Self::Output {
        unimplemented!("Polynomial multiplication is not implemented")
    }
}

impl<R: Semiring, const DEGREE: usize> AddAssign for DensePolynomial<R, DEGREE> {
    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self += &rhs;
    }
}

/// Sealed trait for internal coefficient addition with SIMD specialization
///
/// # SIMD Optimization for i32
///
/// On Aarch64 (Apple Silicon) with NEON enabled, DensePolynomial<i32, _> addition
/// uses ARM NEON SIMD instructions for optimal performance:
/// - Processes 4 i32 elements per iteration using vaddq_s32
/// - Automatically handles any remainder elements with scalar addition
/// - Requires compilation with target features enabled
///
/// ## Requirements
/// - Target: aarch64-apple-darwin (Apple Silicon)
/// - Rust: nightly (for min_specialization feature)
/// - Target feature: +neon (usually enabled by default on Apple Silicon)
///
/// ## Build Instructions
/// To build with SIMD optimizations:
/// ```bash
/// RUSTFLAGS="-C target-cpu=native" cargo +nightly build --release
/// ```
mod sealed {
    use super::*;

    pub trait CoeffAddImpl<const DEGREE: usize>: Sized {
        fn add_coeffs_inplace(lhs: &mut [Self; DEGREE], rhs: &[Self; DEGREE]);
    }

    // Default scalar implementation for all Semiring types
    impl<R: Semiring, const DEGREE: usize> CoeffAddImpl<DEGREE> for R {
        #[allow(clippy::arithmetic_side_effects)]
        #[inline(always)]
        default fn add_coeffs_inplace(lhs: &mut [Self; DEGREE], rhs: &[Self; DEGREE]) {
            for i in 0..DEGREE {
                lhs[i] += &rhs[i];
            }
        }
    }

    // SIMD-optimized implementation for i32 on Aarch64
    // Uses ARM NEON intrinsics to process 4 i32 values in parallel
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    impl<const DEGREE: usize> CoeffAddImpl<DEGREE> for i32 {
        #[allow(clippy::arithmetic_side_effects)]
        #[inline(always)]
        fn add_coeffs_inplace(lhs: &mut [Self; DEGREE], rhs: &[Self; DEGREE]) {
            use std::arch::aarch64::*;
            unsafe {
                let mut i = 0;
                // Process 4 i32 elements at a time using NEON SIMD
                while i + 4 <= DEGREE {
                    let a = vld1q_s32(lhs.as_ptr().add(i));
                    let b = vld1q_s32(rhs.as_ptr().add(i));
                    let sum = vaddq_s32(a, b);
                    vst1q_s32(lhs.as_mut_ptr().add(i), sum);
                    i += 4;
                }

                // Handle remaining elements with scalar addition
                while i < DEGREE {
                    lhs[i] += rhs[i];
                    i += 1;
                }
            }
        }
    }
}

impl<'a, R: Semiring, const DEGREE: usize> AddAssign<&'a Self> for DensePolynomial<R, DEGREE>
where
    R: sealed::CoeffAddImpl<DEGREE>,
{
    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn add_assign(&mut self, rhs: &'a Self) {
        self.coeff_0 += &rhs.coeff_0;
        for i in 0..DEGREE {
            self.coeffs[i] += &rhs.coeffs[i];
        }
    }
}

impl<R: Semiring, const DEGREE: usize> SubAssign for DensePolynomial<R, DEGREE> {
    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self -= &rhs;
    }
}

impl<'a, R: Semiring, const DEGREE: usize> SubAssign<&'a Self> for DensePolynomial<R, DEGREE> {
    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &'a Self) {
        self.coeff_0 -= &rhs.coeff_0;
        for i in 0..DEGREE {
            self.coeffs[i] -= &rhs.coeffs[i];
        }
    }
}

impl<R: Semiring, const DEGREE: usize> MulAssign for DensePolynomial<R, DEGREE> {
    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self *= &rhs;
    }
}

impl<'a, R: Semiring, const DEGREE: usize> MulAssign<&'a Self> for DensePolynomial<R, DEGREE> {
    fn mul_assign(&mut self, _rhs: &'a Self) {
        unimplemented!("Polynomial multiplication is not implemented")
    }
}

impl<R: Ring, const DEGREE: usize> CheckedNeg for DensePolynomial<R, DEGREE> {
    fn checked_neg(&self) -> Option<Self> {
        let coeffs: Option<Vec<R>> = self.coeffs.iter().map(|c| c.checked_neg()).collect();
        Some(Self {
            coeff_0: self.coeff_0.checked_neg()?,
            coeffs: coeffs?.try_into().ok()?,
        })
    }
}

impl<R: Semiring, const DEGREE: usize> CheckedAdd for DensePolynomial<R, DEGREE> {
    fn checked_add(&self, other: &Self) -> Option<Self> {
        let coeffs: Option<Vec<R>> = self
            .coeffs
            .iter()
            .zip(other.coeffs.iter())
            .map(|(a, b)| a.checked_add(b))
            .collect();
        Some(Self {
            coeff_0: self.coeff_0.checked_add(&other.coeff_0)?,
            coeffs: coeffs?.try_into().ok()?,
        })
    }
}

impl<R: Semiring, const DEGREE: usize> CheckedSub for DensePolynomial<R, DEGREE> {
    fn checked_sub(&self, other: &Self) -> Option<Self> {
        let coeffs: Option<Vec<R>> = self
            .coeffs
            .iter()
            .zip(other.coeffs.iter())
            .map(|(a, b)| a.checked_sub(b))
            .collect();
        Some(Self {
            coeff_0: self.coeff_0.checked_sub(&other.coeff_0)?,
            coeffs: coeffs?.try_into().ok()?,
        })
    }
}

impl<R: Semiring, const DEGREE: usize> CheckedMul for DensePolynomial<R, DEGREE> {
    fn checked_mul(&self, _other: &Self) -> Option<Self> {
        unimplemented!("Polynomial multiplication is not implemented")
    }
}

impl<R: Semiring, const DEGREE: usize> Sum for DensePolynomial<R, DEGREE> {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let Some(first) = iter.next() else {
            panic!("Sum of an empty iterator is not defined for DensePolynomial");
        };
        iter.fold(first, |acc, x| {
            acc.checked_add(&x).expect("overflow in sum")
        })
    }
}

impl<'a, R: Semiring, const DEGREE: usize> Sum<&'a Self> for DensePolynomial<R, DEGREE> {
    fn sum<I: Iterator<Item = &'a Self>>(mut iter: I) -> Self {
        let Some(first) = iter.next() else {
            panic!("Sum of an empty iterator is not defined for DensePolynomial");
        };
        iter.fold(first.clone(), |acc, x| {
            acc.checked_add(x).expect("overflow in sum")
        })
    }
}

impl<R: Semiring, const DEGREE: usize> Product for DensePolynomial<R, DEGREE> {
    fn product<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let Some(first) = iter.next() else {
            panic!("Product of an empty iterator is not defined for DensePolynomial");
        };
        iter.fold(first, |acc, x| {
            acc.checked_mul(&x).expect("overflow in product")
        })
    }
}

impl<'a, R: Semiring, const DEGREE: usize> Product<&'a Self> for DensePolynomial<R, DEGREE> {
    fn product<I: Iterator<Item = &'a Self>>(mut iter: I) -> Self {
        let Some(first) = iter.next() else {
            panic!("Product of an empty iterator is not defined for DensePolynomial");
        };
        iter.fold(first.clone(), |acc, x| {
            acc.checked_mul(x).expect("overflow in product")
        })
    }
}

impl<R: Semiring, const DEGREE: usize> Semiring for DensePolynomial<R, DEGREE> {}

impl<R: Ring, const DEGREE: usize> Ring for DensePolynomial<R, DEGREE> {}

impl<R: Semiring, const DEGREE: usize> Distribution<DensePolynomial<R, DEGREE>> for StandardUniform
where
    StandardUniform: Distribution<R>,
    StandardUniform: Distribution<[R; DEGREE]>, // This one we get for free
{
    fn sample<Gen: Rng + ?Sized>(&self, rng: &mut Gen) -> DensePolynomial<R, DEGREE> {
        let coeff_0: R = rng.random();
        let coeffs: [R; DEGREE] = rng.random();
        DensePolynomial { coeff_0, coeffs }
    }
}

//
// Zip-specific traits
//

impl<R: Semiring, C, const DEGREE: usize> EvaluatablePolynomial<C, R> for DensePolynomial<R, DEGREE>
where
    R: for<'a> MulByScalar<&'a C>,
{
    fn evaluate_at_point(&self, point: &[C]) -> Result<R, EvaluationError>
    where
        R: for<'a> MulByScalar<&'a C>,
    {
        if point.len() != DEGREE {
            return Err(EvaluationError::WrongPointWidth {
                expected: DEGREE,
                actual: point.len(),
            });
        }

        // A trivial implementation of a polynomial evaluation at a given point.
        let mut result = self.coeff_0.clone();
        for (coeff, scalar) in self.coeffs.iter().zip(point.iter()) {
            let term = coeff
                .mul_by_scalar(scalar)
                .ok_or(EvaluationError::Overflow)?;
            result = result.checked_add(&term).ok_or(EvaluationError::Overflow)?;
        }
        Ok(result)
    }
}

impl<R: Semiring + ConstTranscribable, const DEGREE: usize> ConstCoeffBitWidth
    for DensePolynomial<R, DEGREE>
{
    const COEFF_BIT_WIDTH: usize = R::NUM_BITS;
}

impl<R: Semiring + Named, const DEGREE: usize> Named for DensePolynomial<R, DEGREE> {
    fn type_name() -> String {
        format!("Poly<{}, {DEGREE}>", R::type_name())
    }
}

impl<R: ConstTranscribable + Default, const DEGREE: usize> ConstTranscribable
    for DensePolynomial<R, DEGREE>
{
    const NUM_BYTES: usize = R::NUM_BYTES * (DEGREE + 1);

    #[allow(clippy::arithmetic_side_effects)]
    fn read_transcription_bytes(bytes: &[u8]) -> Self {
        assert_eq!(
            bytes.len(),
            R::NUM_BYTES * (DEGREE + 1),
            "Invalid byte length for DensePolynomial: expected {}, got {}",
            R::NUM_BYTES * (DEGREE + 1),
            bytes.len()
        );

        let coeff_0 = R::read_transcription_bytes(&bytes[0..R::NUM_BYTES]);

        let mut coeffs = array::from_fn::<_, DEGREE, _>(|_| R::default());
        for i in 1..=DEGREE {
            let start = i * R::NUM_BYTES;
            let end = start + R::NUM_BYTES;
            coeffs[i - 1] = R::read_transcription_bytes(&bytes[start..end]);
        }

        Self { coeff_0, coeffs }
    }

    fn write_transcription_bytes(&self, buf: &mut [u8]) {
        for (chunk, coeff) in buf
            .chunks_exact_mut(R::NUM_BYTES)
            .zip(iter::once(&self.coeff_0).chain(self.coeffs.iter()))
        {
            coeff.write_transcription_bytes(chunk);
        }
    }
}

impl<R, S, const DEGREE: usize> FromRef<DensePolynomial<S, DEGREE>> for DensePolynomial<R, DEGREE>
where
    R: Semiring + FromRef<S> + Default,
{
    fn from_ref(value: &DensePolynomial<S, DEGREE>) -> Self {
        let coeff_0 = R::from_ref(&value.coeff_0);
        let mut coeffs = array::from_fn::<_, DEGREE, _>(|_| R::default());
        coeffs
            .iter_mut()
            .zip(value.coeffs.iter())
            .for_each(|(coeff, other_coeff)| {
                *coeff = R::from_ref(other_coeff);
            });
        DensePolynomial { coeff_0, coeffs }
    }
}

impl<R, S, const DEGREE: usize> From<&DensePolynomial<S, DEGREE>> for DensePolynomial<R, DEGREE>
where
    R: Semiring + FromRef<S> + Default,
{
    fn from(value: &DensePolynomial<S, DEGREE>) -> Self {
        Self::from_ref(value)
    }
}

impl<'a, R, S, const DEGREE: usize> MulByScalar<&'a S> for DensePolynomial<R, DEGREE>
where
    R: Semiring + MulByScalar<&'a S>,
{
    fn mul_by_scalar(&self, rhs: &'a S) -> Option<Self> {
        let coeff_0 = self.coeff_0.mul_by_scalar(rhs)?;
        let coeffs: Option<Vec<R>> = self.coeffs.iter().map(|c| c.mul_by_scalar(rhs)).collect();

        Some(Self {
            coeff_0,
            coeffs: coeffs?.try_into().ok()?,
        })
    }
}

impl<R, F, const DEGREE: usize> ProjectableToField<F> for DensePolynomial<R, DEGREE>
where
    R: Semiring,
    F: PrimeField + for<'a> FromWithConfig<&'a R> + for<'a> MulByScalar<&'a F> + 'static,
{
    #![allow(clippy::arithmetic_side_effects)] // False alert, field operations are safe
    fn prepare_projection(sampled_value: &F) -> impl Fn(&Self) -> F + 'static {
        let mut r_powers = vec![sampled_value.clone(); DEGREE];
        for i in 1..DEGREE {
            r_powers[i] = r_powers[i - 1].clone() * sampled_value;
        }
        let field_cfg = sampled_value.cfg().clone();

        move |poly: &Self| {
            let coeff_0 = (&poly.coeff_0).into_with_cfg(&field_cfg);
            let coeffs: [F; DEGREE] = poly
                .coeffs
                .iter()
                .map(|v| v.into_with_cfg(&field_cfg))
                .collect_array()
                .expect("unreachable");

            let poly2 = DensePolynomial { coeff_0, coeffs };
            poly2
                .evaluate_at_point(&r_powers)
                .expect("Failed to evaluate polynomial at point")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_i32_addition() {
        // Test with various sizes to ensure addition works correctly
        
        // Test with 4 elements
        let poly1: DensePolynomial<i32, 4> = DensePolynomial {
            coeff_0: 1,
            coeffs: [2, 3, 4, 5],
        };
        let poly2: DensePolynomial<i32, 4> = DensePolynomial {
            coeff_0: 10,
            coeffs: [20, 30, 40, 50],
        };
        let mut result = poly1.clone();
        result += &poly2;
        assert_eq!(result.coeff_0, 11);
        assert_eq!(result.coeffs, [22, 33, 44, 55]);

        // Test with 8 elements
        let poly3: DensePolynomial<i32, 8> = DensePolynomial {
            coeff_0: 1,
            coeffs: [1, 2, 3, 4, 5, 6, 7, 8],
        };
        let poly4: DensePolynomial<i32, 8> = DensePolynomial {
            coeff_0: 100,
            coeffs: [10, 20, 30, 40, 50, 60, 70, 80],
        };
        let mut result2 = poly3.clone();
        result2 += &poly4;
        assert_eq!(result2.coeff_0, 101);
        assert_eq!(result2.coeffs, [11, 22, 33, 44, 55, 66, 77, 88]);

        // Test with 10 elements
        let poly5: DensePolynomial<i32, 10> = DensePolynomial {
            coeff_0: -5,
            coeffs: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        };
        let poly6: DensePolynomial<i32, 10> = DensePolynomial {
            coeff_0: 15,
            coeffs: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        };
        let mut result3 = poly5.clone();
        result3 += &poly6;
        assert_eq!(result3.coeff_0, 10);
        assert_eq!(result3.coeffs, [11, 22, 33, 44, 55, 66, 77, 88, 99, 110]);

        // Test with 3 elements
        let poly7: DensePolynomial<i32, 3> = DensePolynomial {
            coeff_0: 7,
            coeffs: [1, 2, 3],
        };
        let poly8: DensePolynomial<i32, 3> = DensePolynomial {
            coeff_0: 3,
            coeffs: [4, 5, 6],
        };
        let mut result4 = poly7.clone();
        result4 += &poly8;
        assert_eq!(result4.coeff_0, 10);
        assert_eq!(result4.coeffs, [5, 7, 9]);
    }

    #[test]
    fn test_non_i32_addition_still_works() {
        // Ensure other types still work correctly with the generic implementation
        let poly1: DensePolynomial<i64, 4> = DensePolynomial {
            coeff_0: 1i64,
            coeffs: [2, 3, 4, 5],
        };
        let poly2: DensePolynomial<i64, 4> = DensePolynomial {
            coeff_0: 10,
            coeffs: [20, 30, 40, 50],
        };
        let mut result = poly1.clone();
        result += &poly2;
        assert_eq!(result.coeff_0, 11);
        assert_eq!(result.coeffs, [22, 33, 44, 55]);
    }
}
