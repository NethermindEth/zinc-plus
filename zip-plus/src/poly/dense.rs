use std::fmt::Display;
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};
use num_traits::{One, CheckedAdd, CheckedMul, CheckedNeg, CheckedSub, ConstZero, Zero};
use crypto_primitives::{Ring};

pub trait Polynomial<R: Ring> {
    /// Returns the degree of the polynomial - a number of coefficients.
    fn degree(&self) -> usize;

    /// Coefficients of the polynomial, lowest degree first.
    fn coeffs(&self) -> &[R];
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DensePolynomial<R: Ring> {
    /// Coefficients of the polynomial, lowest degree first.
    coeffs: Vec<R>,
}

impl<R: Ring> DensePolynomial<R> {
    pub fn new(coeffs: Vec<R>) -> Self {
        // Drop trailing zero coefficients
        let last_non_zero = coeffs.iter().rposition(|c| !c.is_zero()).unwrap_or(0);
        let coeffs = coeffs.into_iter().take(last_non_zero + 1).collect();
        DensePolynomial { coeffs }
    }
}

impl<R: Ring> Polynomial<R> for DensePolynomial<R> {
    fn degree(&self) -> usize {
        self.coeffs.len() - 1
    }

    fn coeffs(&self) -> &[R] {
        &self.coeffs
    }
}

impl<R: Ring> Display for DensePolynomial<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, coeff) in self.coeffs.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", coeff)?;
        }
        write!(f, "]")?;
        Ok(())
    }
}

impl<R: Ring> ConstZero for DensePolynomial<R> {
    const ZERO: Self = Self { coeffs: Vec::new() };
}

impl<R: Ring> Zero for DensePolynomial<R> {
    fn zero() -> Self {
        Self::ZERO
    }

    fn is_zero(&self) -> bool {
        self == &Self::ZERO
    }
}

impl<R: Ring> One for DensePolynomial<R> {
    fn one() -> Self {
        Self { coeffs: vec![R::one()] }
    }
}

impl<R: Ring> Add for DensePolynomial<R> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self + &rhs
    }
}

impl<'a, R: Ring> Add<&'a Self> for DensePolynomial<R> {
    type Output = Self;

    fn add(self, rhs: &'a Self) -> Self::Output {
        let max_len = self.coeffs.len().max(rhs.coeffs.len());
        let mut result = Vec::with_capacity(max_len);

        for i in 0..max_len {
            let a = if i < self.coeffs.len() { self.coeffs[i].clone() } else { R::zero() };
            let b = if i < rhs.coeffs.len() { rhs.coeffs[i].clone() } else { R::zero() };
            result.push(a + &b);
        }

        Self::new(result)
    }
}

impl<R: Ring> Sub for DensePolynomial<R> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self - &rhs
    }
}

impl<'a, R: Ring> Sub<&'a Self> for DensePolynomial<R> {
    type Output = Self;

    fn sub(self, rhs: &'a Self) -> Self::Output {
        let max_len = self.coeffs.len().max(rhs.coeffs.len());
        let mut result = Vec::with_capacity(max_len);

        for i in 0..max_len {
            let a = if i < self.coeffs.len() { self.coeffs[i].clone() } else { R::zero() };
            let b = if i < rhs.coeffs.len() { rhs.coeffs[i].clone() } else { R::zero() };
            result.push(a - &b);
        }

        Self::new(result)
    }
}

impl<R: Ring> Mul for DensePolynomial<R> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self * &rhs
    }
}

impl<'a, R: Ring> Mul<&'a Self> for DensePolynomial<R> {
    type Output = Self;

    fn mul(self, _rhs: &'a Self) -> Self::Output {
        unimplemented!("Polynomial multiplication is not implemented")
    }
}

impl<R: Ring> AddAssign for DensePolynomial<R> {
    fn add_assign(&mut self, rhs: Self) {
        *self += &rhs;
    }
}

impl<'a, R: Ring> AddAssign<&'a Self> for DensePolynomial<R> {
    fn add_assign(&mut self, rhs: &'a Self) {
        let max_len = self.coeffs.len().max(rhs.coeffs.len());
        self.coeffs.resize_with(max_len, R::zero);

        for (i, b) in rhs.coeffs.iter().enumerate() {
            self.coeffs[i] += b;
        }

        // Normalize by removing trailing zeros
        let last_non_zero = self.coeffs.iter().rposition(|c| !c.is_zero()).unwrap_or(0);
        self.coeffs.truncate(last_non_zero + 1);
    }
}

impl<R: Ring> SubAssign for DensePolynomial<R> {
    fn sub_assign(&mut self, rhs: Self) {
        *self -= &rhs;
    }
}

impl<'a, R: Ring> SubAssign<&'a Self> for DensePolynomial<R> {
    fn sub_assign(&mut self, rhs: &'a Self) {
        let max_len = self.coeffs.len().max(rhs.coeffs.len());
        self.coeffs.resize_with(max_len, R::zero);

        for (i, b) in rhs.coeffs.iter().enumerate() {
            self.coeffs[i] -= b;
        }

        // Normalize by removing trailing zeros
        let last_non_zero = self.coeffs.iter().rposition(|c| !c.is_zero()).unwrap_or(0);
        self.coeffs.truncate(last_non_zero + 1);
    }
}

impl<R: Ring> MulAssign for DensePolynomial<R> {
    fn mul_assign(&mut self, rhs: Self) {
        *self *= &rhs;
    }
}

impl<'a, R: Ring> MulAssign<&'a Self> for DensePolynomial<R> {
    fn mul_assign(&mut self, _rhs: &'a Self) {
        unimplemented!("Polynomial multiplication is not implemented")
    }
}

impl<R: Ring> CheckedNeg for DensePolynomial<R> {
    fn checked_neg(&self) -> Option<Self> {
        let mut result = Vec::with_capacity(self.coeffs.len());
        
        for coeff in &self.coeffs {
            result.push(coeff.checked_neg()?);
        }
        
        Some(Self::new(result))
    }
}

impl<R: Ring> CheckedAdd for DensePolynomial<R> {
    fn checked_add(&self, other: &Self) -> Option<Self> {
        let max_len = self.coeffs.len().max(other.coeffs.len());
        let mut result = Vec::with_capacity(max_len);
        
        for i in 0..max_len {
            let a = if i < self.coeffs.len() { &self.coeffs[i] } else { &R::zero() };
            let b = if i < other.coeffs.len() { &other.coeffs[i] } else { &R::zero() };
            result.push(a.checked_add(b)?);
        }
        
        Some(Self::new(result))
    }
}

impl<R: Ring> CheckedSub for DensePolynomial<R> {
    fn checked_sub(&self, other: &Self) -> Option<Self> {
        let max_len = self.coeffs.len().max(other.coeffs.len());
        let mut result = Vec::with_capacity(max_len);
        
        for i in 0..max_len {
            let a = if i < self.coeffs.len() { &self.coeffs[i] } else { &R::zero() };
            let b = if i < other.coeffs.len() { &other.coeffs[i] } else { &R::zero() };
            result.push(a.checked_sub(b)?);
        }
        
        Some(Self::new(result))
    }
}

impl<R: Ring> CheckedMul for DensePolynomial<R> {
    fn checked_mul(&self, _other: &Self) -> Option<Self> {
        unimplemented!("Polynomial multiplication is not implemented")
    }
}

impl<R: Ring> Sum for DensePolynomial<R> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| {
            acc.checked_add(&x).expect("overflow in sum")
        })
    }
}

impl<'a, R: Ring> Sum<&'a Self> for DensePolynomial<R> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| {
            acc.checked_add(x).expect("overflow in sum")
        })
    }
}

impl<R: Ring> Product for DensePolynomial<R> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| {
            acc.checked_mul(&x).expect("overflow in product")
        })
    }
}

impl<'a, R: Ring> Product<&'a Self> for DensePolynomial<R> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| {
            acc.checked_mul(x).expect("overflow in product")
        })
    }
}

impl<R: Ring> Ring for DensePolynomial<R> {}
