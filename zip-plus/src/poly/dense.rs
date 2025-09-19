use crate::{
    pcs::structs::{AsPackable, MulByScalar, PackedInt},
    traits::Transcribable,
    utils::ReinterpretVector,
};
use crypto_primitives::{Ring, crypto_bigint_int::Int};
use num_traits::{CheckedAdd, CheckedMul, CheckedNeg, CheckedSub, One, Zero};
use p3_field::Packable;
use std::{
    array,
    fmt::Display,
    iter,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

pub trait Polynomial<R: Ring> {
    /// Returns the degree of the polynomial - a number of coefficients.
    fn degree(&self) -> usize;

    /// Coefficients of the polynomial, lowest degree first.
    fn coeffs<'a>(&'a self) -> impl Iterator<Item = &'a R>
    where
        R: 'a;
}

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

impl<R: Ring, const DEGREE: usize> DensePolynomial<R, DEGREE> {
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
}

impl<R: Ring, const DEGREE: usize> Polynomial<R> for DensePolynomial<R, DEGREE> {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn coeffs<'a>(&'a self) -> impl Iterator<Item = &'a R>
    where
        R: 'a,
    {
        iter::once(&self.coeff_0).chain(self.coeffs.iter())
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

impl<R: Ring, const DEGREE: usize> Zero for DensePolynomial<R, DEGREE> {
    fn zero() -> Self {
        Self {
            coeff_0: R::zero(),
            coeffs: array::from_fn::<_, DEGREE, _>(|_| R::zero()),
        }
    }

    fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|c| c.is_zero())
    }
}

impl<R: Ring, const DEGREE: usize> One for DensePolynomial<R, DEGREE> {
    fn one() -> Self {
        Self {
            coeff_0: R::one(),
            coeffs: array::from_fn::<_, DEGREE, _>(|_| R::zero()),
        }
    }
}

impl<R: Ring, const DEGREE: usize> Add for DensePolynomial<R, DEGREE> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        self + &rhs
    }
}

impl<'a, R: Ring, const DEGREE: usize> Add<&'a Self> for DensePolynomial<R, DEGREE> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn add(mut self, rhs: &'a Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<R: Ring, const DEGREE: usize> Sub for DensePolynomial<R, DEGREE> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        self - &rhs
    }
}

impl<'a, R: Ring, const DEGREE: usize> Sub<&'a Self> for DensePolynomial<R, DEGREE> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn sub(mut self, rhs: &'a Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl<R: Ring, const DEGREE: usize> Mul for DensePolynomial<R, DEGREE> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        self * &rhs
    }
}

impl<'a, R: Ring, const DEGREE: usize> Mul<&'a Self> for DensePolynomial<R, DEGREE> {
    type Output = Self;

    fn mul(self, _rhs: &'a Self) -> Self::Output {
        unimplemented!("Polynomial multiplication is not implemented")
    }
}

impl<R: Ring, const DEGREE: usize> AddAssign for DensePolynomial<R, DEGREE> {
    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self += &rhs;
    }
}

impl<'a, R: Ring, const DEGREE: usize> AddAssign<&'a Self> for DensePolynomial<R, DEGREE> {
    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn add_assign(&mut self, rhs: &'a Self) {
        self.coeff_0 += &rhs.coeff_0;
        for i in 0..=DEGREE {
            self.coeffs[i] += &rhs.coeffs[i];
        }
    }
}

impl<R: Ring, const DEGREE: usize> SubAssign for DensePolynomial<R, DEGREE> {
    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self -= &rhs;
    }
}

impl<'a, R: Ring, const DEGREE: usize> SubAssign<&'a Self> for DensePolynomial<R, DEGREE> {
    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &'a Self) {
        self.coeff_0 -= &rhs.coeff_0;
        for i in 0..=DEGREE {
            self.coeffs[i] -= &rhs.coeffs[i];
        }
    }
}

impl<R: Ring, const DEGREE: usize> MulAssign for DensePolynomial<R, DEGREE> {
    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self *= &rhs;
    }
}

impl<'a, R: Ring, const DEGREE: usize> MulAssign<&'a Self> for DensePolynomial<R, DEGREE> {
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

impl<R: Ring, const DEGREE: usize> CheckedAdd for DensePolynomial<R, DEGREE> {
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

impl<R: Ring, const DEGREE: usize> CheckedSub for DensePolynomial<R, DEGREE> {
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

impl<R: Ring, const DEGREE: usize> CheckedMul for DensePolynomial<R, DEGREE> {
    fn checked_mul(&self, _other: &Self) -> Option<Self> {
        unimplemented!("Polynomial multiplication is not implemented")
    }
}

impl<R: Ring, const DEGREE: usize> Sum for DensePolynomial<R, DEGREE> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| {
            acc.checked_add(&x).expect("overflow in sum")
        })
    }
}

impl<'a, R: Ring, const DEGREE: usize> Sum<&'a Self> for DensePolynomial<R, DEGREE> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| {
            acc.checked_add(x).expect("overflow in sum")
        })
    }
}

impl<R: Ring, const DEGREE: usize> Product for DensePolynomial<R, DEGREE> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| {
            acc.checked_mul(&x).expect("overflow in product")
        })
    }
}

impl<'a, R: Ring, const DEGREE: usize> Product<&'a Self> for DensePolynomial<R, DEGREE> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| {
            acc.checked_mul(x).expect("overflow in product")
        })
    }
}

impl<R: Ring, const DEGREE: usize> Ring for DensePolynomial<R, DEGREE> {}

//
// Zip-specific traits
//

impl<R: Transcribable + Default, const DEGREE: usize> Transcribable for DensePolynomial<R, DEGREE> {
    const NUM_BYTES: usize = R::NUM_BYTES * (DEGREE + 1);

    #[allow(clippy::arithmetic_side_effects)]
    fn from_transcription_bytes(bytes: &[u8]) -> Self {
        assert_eq!(
            bytes.len(),
            R::NUM_BYTES * (DEGREE + 1),
            "Invalid byte length for DensePolynomial: expected {}, got {}",
            R::NUM_BYTES * (DEGREE + 1),
            bytes.len()
        );

        let coeff_0 = R::from_transcription_bytes(&bytes[0..R::NUM_BYTES]);

        let mut coeffs = array::from_fn::<_, DEGREE, _>(|_| R::default());
        for i in 1..=DEGREE {
            let start = i * R::NUM_BYTES;
            let end = start + R::NUM_BYTES;
            coeffs[i - 1] = R::from_transcription_bytes(&bytes[start..end]);
        }

        Self { coeff_0, coeffs }
    }

    fn to_transcription_bytes(&self, buf: &mut [u8]) {
        for (chunk, coeff) in buf
            .chunks_exact_mut(R::NUM_BYTES)
            .zip(iter::once(&self.coeff_0).chain(self.coeffs.iter()))
        {
            coeff.to_transcription_bytes(chunk);
        }
    }
}

impl<'a, const LIMBS: usize, const LIMBS2: usize, const DEGREE: usize>
    From<&'a DensePolynomial<Int<LIMBS2>, DEGREE>> for DensePolynomial<Int<LIMBS>, DEGREE>
{
    fn from(value: &'a DensePolynomial<Int<LIMBS2>, DEGREE>) -> Self {
        if LIMBS < LIMBS2 {
            panic!("Cannot convert polynomial of Int<{LIMBS}> to smaller Int<{LIMBS2}>");
        }
        let coeff_0 = value.coeff_0.resize();
        let mut coeffs = [Int::<LIMBS>::default(); DEGREE];
        coeffs
            .iter_mut()
            .zip(value.coeffs.iter())
            .for_each(|(coeff, other_coeff)| {
                *coeff = other_coeff.resize();
            });
        DensePolynomial { coeff_0, coeffs }
    }
}

impl<'a, const LIMBS: usize, const LIMBS2: usize, const DEGREE: usize> MulByScalar<&'a Int<LIMBS2>>
    for DensePolynomial<Int<LIMBS>, DEGREE>
{
    fn mul_by_scalar(&self, rhs: &'a Int<LIMBS2>) -> Option<Self> {
        if LIMBS < LIMBS2 {
            return None;
        }
        let coeff_0 = self.coeff_0.mul_by_scalar(rhs)?;
        let coeffs: Option<Vec<Int<LIMBS>>> =
            self.coeffs.iter().map(|c| c.mul_by_scalar(rhs)).collect();

        Some(Self {
            coeff_0,
            coeffs: coeffs?.try_into().ok()?,
        })
    }
}

//
// PackableDensePolynomial
//

impl<const LIMBS: usize, const DEGREE: usize> AsPackable for DensePolynomial<Int<LIMBS>, DEGREE> {
    type Packable = PackableDensePolynomial<PackedInt<LIMBS>, DEGREE>;
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct PackableDensePolynomial<R, const DEGREE: usize>(DensePolynomial<R, DEGREE>);

impl<R: Copy, const DEGREE: usize> Copy for PackableDensePolynomial<R, DEGREE> {}

impl<R: Packable, const DEGREE: usize> Packable for PackableDensePolynomial<R, DEGREE> {}

unsafe impl<R: AsPackable, const DEGREE: usize>
    ReinterpretVector<PackableDensePolynomial<R::Packable, DEGREE>> for DensePolynomial<R, DEGREE>
{
}

impl<R: Transcribable + Default, const DEGREE: usize> Transcribable
    for PackableDensePolynomial<R, DEGREE>
{
    const NUM_BYTES: usize = DensePolynomial::<R, DEGREE>::NUM_BYTES;

    fn from_transcription_bytes(bytes: &[u8]) -> Self {
        Self(DensePolynomial::from_transcription_bytes(bytes))
    }

    fn to_transcription_bytes(&self, buf: &mut [u8]) {
        self.0.to_transcription_bytes(buf)
    }
}
