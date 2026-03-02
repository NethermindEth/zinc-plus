use crypto_primitives::{PrimeField, Semiring};
use derive_more::From;
use num_traits::{CheckedAdd, CheckedMul, CheckedNeg, CheckedSub, ConstZero, Euclid, Zero};
use std::{
    fmt::Display,
    ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Rem, Sub, SubAssign},
};
use zinc_utils::UNCHECKED;

use crate::{
    EvaluatablePolynomial, EvaluationError, Polynomial,
    univariate::{dense::DensePolynomial, dynamic},
};

/// Polynomials of dynamic degree. The implementation
/// is tailored to work with random finite fields.
/// To be used in UAIR and PIOP where ZIP+ degree bound
/// is not observed anymore.
///
/// Note that operations involving dynamic polynomials
/// do not trim leading zeros meaning
/// one can end up with unequal objects of the type
/// `DynamicPoly<F>` that represent equal polynomials,
/// therefore `trim` has to be called before checking
/// equality.
#[derive(Debug, Clone, From, Hash, PartialEq, Eq)]
pub struct DynamicPolynomialF<F: PrimeField> {
    pub coeffs: Vec<F>,
}

impl<F: PrimeField> DynamicPolynomialF<F> {
    /// Create a new polynomial with the given coefficients.
    #[inline(always)]
    pub fn new_trimmed(coeffs: impl AsRef<[F]>) -> Self {
        Self {
            coeffs: dynamic::new_coeffs_trimmed(coeffs.as_ref(), F::is_zero),
        }
    }

    #[inline(always)]
    pub fn new(coeffs: impl AsRef<[F]>) -> Self {
        Self {
            coeffs: Vec::from(coeffs.as_ref()),
        }
    }

    #[inline(always)]
    pub fn degree(&self) -> Option<usize> {
        dynamic::degree(&self.coeffs, F::is_zero)
    }

    #[inline(always)]
    pub fn trim(&mut self) {
        dynamic::trim(&mut self.coeffs, F::is_zero);
    }

    pub fn constant_poly(a: F) -> Self {
        if F::is_zero(&a) {
            Self::default()
        } else {
            DynamicPolynomialF { coeffs: vec![a] }
        }
    }

    /// No bounds check — caller must ensure `powers.len() >= self.coeffs.len()`.
    #[inline(always)]
    #[allow(clippy::arithmetic_side_effects)]
    pub fn dot_with_powers(&self, powers: &[F], zero: F) -> F {
        self.coeffs
            .iter()
            .zip(powers)
            .fold(zero, |acc, (coeff, power)| acc + coeff.clone() * power)
    }
}

impl<F: PrimeField> Display for DynamicPolynomialF<F> {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        dynamic::display(&self.coeffs, f)
    }
}

impl<F: PrimeField> Default for DynamicPolynomialF<F> {
    fn default() -> Self {
        Self {
            coeffs: Default::default(),
        }
    }
}

impl<F: PrimeField> Zero for DynamicPolynomialF<F> {
    #[inline(always)]
    fn zero() -> Self {
        Default::default()
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        dynamic::is_zero(&self.coeffs, F::is_zero)
    }
}

impl<F: PrimeField> ConstZero for DynamicPolynomialF<F> {
    const ZERO: Self = Self { coeffs: Vec::new() };
}

impl<F: PrimeField> Neg for DynamicPolynomialF<F> {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self {
            coeffs: dynamic::neg(self.coeffs),
        }
    }
}

impl<F: PrimeField> Add for DynamicPolynomialF<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.add(&rhs)
    }
}

impl<F: PrimeField> Add<&Self> for DynamicPolynomialF<F> {
    type Output = Self;

    #[inline(always)]
    fn add(mut self, rhs: &Self) -> Self::Output {
        self.add_assign(rhs);

        self
    }
}

impl<F: PrimeField> Sub for DynamicPolynomialF<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.sub(&rhs)
    }
}

impl<F: PrimeField> Sub<&Self> for DynamicPolynomialF<F> {
    type Output = Self;

    fn sub(mut self, rhs: &Self) -> Self::Output {
        self.sub_assign(rhs);

        self
    }
}

impl<F: PrimeField> Mul for DynamicPolynomialF<F> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

impl<F: PrimeField> Mul<&Self> for DynamicPolynomialF<F> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    fn mul(self, rhs: &Self) -> Self::Output {
        &self * rhs
    }
}

impl<'a, F: PrimeField> Mul<&'a DynamicPolynomialF<F>> for &'a DynamicPolynomialF<F> {
    type Output = DynamicPolynomialF<F>;

    #[allow(clippy::arithmetic_side_effects)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::Output {
            coeffs: dynamic::mul::<_, UNCHECKED>(&self.coeffs, &rhs.coeffs, F::is_zero)
                .expect("overflow in a field will not happen"),
        }
    }
}

impl<F: PrimeField> CheckedAdd for DynamicPolynomialF<F> {
    #[allow(clippy::arithmetic_side_effects)] // We are in a field.
    fn checked_add(&self, rhs: &Self) -> Option<Self> {
        Some(self.clone() + rhs)
    }
}

impl<F: PrimeField> CheckedSub for DynamicPolynomialF<F> {
    #[allow(clippy::arithmetic_side_effects)] // We are in a field.
    fn checked_sub(&self, rhs: &Self) -> Option<Self> {
        Some(self.clone() - rhs)
    }
}

impl<F: PrimeField> CheckedMul for DynamicPolynomialF<F> {
    #[allow(clippy::arithmetic_side_effects)] // We are in a field.
    fn checked_mul(&self, rhs: &Self) -> Option<Self> {
        Some(self * rhs)
    }
}

impl<F: PrimeField> CheckedNeg for DynamicPolynomialF<F> {
    fn checked_neg(&self) -> Option<Self> {
        // We are in a field.
        Some(self.clone().neg())
    }
}

impl<F: PrimeField> AddAssign for DynamicPolynomialF<F> {
    fn add_assign(&mut self, rhs: Self) {
        self.add_assign(&rhs);
    }
}

impl<F: PrimeField> AddAssign<&Self> for DynamicPolynomialF<F> {
    #[allow(clippy::arithmetic_side_effects)] // by definition
    fn add_assign(&mut self, rhs: &Self) {
        dynamic::add_assign(&mut self.coeffs, &rhs.coeffs, |elem| {
            F::zero_with_cfg(elem.cfg())
        });
    }
}

impl<F: PrimeField> SubAssign for DynamicPolynomialF<F> {
    #[allow(clippy::arithmetic_side_effects)] // by definition
    fn sub_assign(&mut self, rhs: Self) {
        self.sub_assign(&rhs);
    }
}

impl<F: PrimeField> SubAssign<&Self> for DynamicPolynomialF<F> {
    #[allow(clippy::arithmetic_side_effects)] // by definition
    fn sub_assign(&mut self, rhs: &Self) {
        dynamic::sub_assign(&mut self.coeffs, &rhs.coeffs, |elem| {
            F::zero_with_cfg(elem.cfg())
        });
    }
}

impl<F: PrimeField> MulAssign for DynamicPolynomialF<F> {
    #[allow(clippy::arithmetic_side_effects)] // by definition
    fn mul_assign(&mut self, rhs: Self) {
        let res = rhs * &*self;

        *self = res
    }
}

impl<F: PrimeField> MulAssign<&Self> for DynamicPolynomialF<F> {
    #[allow(clippy::arithmetic_side_effects)] // by definition
    fn mul_assign(&mut self, rhs: &Self) {
        let res = &*self * rhs;

        *self = res;
    }
}

impl<F: PrimeField> Semiring for DynamicPolynomialF<F> {}

impl<F: PrimeField> Div for DynamicPolynomialF<F> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self.div_rem_euclid(&rhs).0
    }
}

impl<F: PrimeField> Rem for DynamicPolynomialF<F> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        self.div_rem_euclid(&rhs).1
    }
}

impl<F: PrimeField> Euclid for DynamicPolynomialF<F> {
    fn div_euclid(&self, v: &Self) -> Self {
        self.div_rem_euclid(v).0
    }

    fn rem_euclid(&self, v: &Self) -> Self {
        self.div_rem_euclid(v).1
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn div_rem_euclid(&self, divisor: &Self) -> (Self, Self) {
        let zero_poly = DynamicPolynomialF::zero();

        let divisor_deg = divisor.degree().expect("division by zero polynomial");

        let Some(dividend_deg) = self.degree() else {
            return (zero_poly.clone(), zero_poly);
        };

        let cfg = self.coeffs[0].cfg();
        let zero = F::zero_with_cfg(cfg);

        if dividend_deg < divisor_deg {
            return (zero_poly.clone(), self.clone());
        }

        // Copy dividend as working remainder
        let mut remainder = self.coeffs.clone();
        let quotient_len = dividend_deg - divisor_deg + 1;
        let mut quotient = vec![zero.clone(); quotient_len];

        let divisor_lead_inv = F::one_with_cfg(cfg) / divisor.coeffs[divisor_deg].clone();

        for i in (0..quotient_len).rev() {
            let rem_idx = i + divisor_deg;
            if remainder[rem_idx] == zero {
                continue;
            }

            // Compute quotient coefficient
            let coeff = remainder[rem_idx].clone() * &divisor_lead_inv;

            // Subtract coeff * x^i * divisor from remainder (skip leading term)
            for j in 0..divisor_deg {
                remainder[i + j] -= coeff.clone() * &divisor.coeffs[j];
            }
            // Leading coefficient is guaranteed to be zero
            remainder[rem_idx] = zero.clone();

            quotient[i] = coeff;
        }

        (
            DynamicPolynomialF { coeffs: quotient },
            DynamicPolynomialF { coeffs: remainder },
        )
    }
}

impl<F: PrimeField, const DEGFEE_PLUS_ONE: usize> From<DensePolynomial<F, DEGFEE_PLUS_ONE>>
    for DynamicPolynomialF<F>
{
    fn from(dense_poly: DensePolynomial<F, DEGFEE_PLUS_ONE>) -> Self {
        Self {
            coeffs: Vec::from(dense_poly.coeffs),
        }
    }
}

impl<F: PrimeField> Polynomial<F> for DynamicPolynomialF<F> {
    const DEGREE_BOUND: usize = usize::MAX;
}

impl<F: PrimeField> EvaluatablePolynomial<F, F> for DynamicPolynomialF<F> {
    type EvaluationPoint = F;

    fn evaluate_at_point(&self, point: &F) -> Result<F, EvaluationError> {
        // Horner's method.
        let mut result = self
            .coeffs
            .last()
            .cloned()
            .unwrap_or(F::zero_with_cfg(point.cfg()));

        for coeff in self.coeffs.iter().rev().skip(1) {
            result *= point;
            result += coeff;
        }

        Ok(result)
    }
}

impl<F: PrimeField> FromIterator<F> for DynamicPolynomialF<F> {
    #[inline(always)]
    fn from_iter<T: IntoIterator<Item = F>>(iter: T) -> Self {
        Self {
            coeffs: iter.into_iter().collect(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects)]
mod tests {
    use crypto_bigint::{Odd, modular::MontyParams};
    use crypto_primitives::{FromWithConfig, crypto_bigint_monty::F256};
    use proptest::prelude::*;
    use rand::Rng;

    use super::*;

    const LIMBS: usize = 4;
    type F = F256;

    fn test_config() -> MontyParams<LIMBS> {
        let modulus = crypto_bigint::Uint::<LIMBS>::from_be_hex(
            "0000000000000000000000000000000000860995AE68FC80E1B1BD1E39D54B33",
        );
        let modulus = Odd::new(modulus).expect("modulus should be odd");
        MontyParams::new(modulus)
    }

    #[test]
    fn new_creates_correctly() {
        let field_cfg = test_config();
        assert_eq!(
            DynamicPolynomialF::new_trimmed([
                F::from_with_cfg(1i32, &field_cfg),
                F::from_with_cfg(2i32, &field_cfg),
                F::from_with_cfg(3i32, &field_cfg),
                F::zero_with_cfg(&field_cfg),
                F::zero_with_cfg(&field_cfg),
            ]),
            DynamicPolynomialF {
                coeffs: vec![
                    F::from_with_cfg(1i32, &field_cfg),
                    F::from_with_cfg(2i32, &field_cfg),
                    F::from_with_cfg(3i32, &field_cfg),
                ]
            }
        );
    }

    fn get_2_test_polynomial() -> (DynamicPolynomialF<F>, DynamicPolynomialF<F>) {
        let field_cfg = test_config();

        let x = DynamicPolynomialF::new(vec![
            F::from_with_cfg(2i32, &field_cfg),
            F::zero_with_cfg(&field_cfg),
            F::from_with_cfg(2i32, &field_cfg),
            F::zero_with_cfg(&field_cfg),
            F::zero_with_cfg(&field_cfg),
        ]);

        let y = DynamicPolynomialF::new(vec![
            F::from_with_cfg(1i32, &field_cfg),
            F::from_with_cfg(2i32, &field_cfg),
            F::from_with_cfg(3i32, &field_cfg),
        ]);

        (x, y)
    }

    #[test]
    fn add_zero() {
        let field_cfg = test_config();

        assert_eq!(
            DynamicPolynomialF::<F>::ZERO + DynamicPolynomialF::ZERO,
            DynamicPolynomialF::ZERO
        );

        let x = DynamicPolynomialF::new(vec![
            F::from_with_cfg(2i32, &field_cfg),
            F::zero_with_cfg(&field_cfg),
            F::from_with_cfg(2i32, &field_cfg),
            F::zero_with_cfg(&field_cfg),
            F::zero_with_cfg(&field_cfg),
        ]);

        assert_eq!(x.clone() + &DynamicPolynomialF::ZERO, x);
        assert_eq!(DynamicPolynomialF::ZERO + x.clone(), x);
        assert_eq!(DynamicPolynomialF::ZERO + &x, x);

        let mut y = x.clone();

        y += DynamicPolynomialF::ZERO;

        assert_eq!(y, x);

        y += &DynamicPolynomialF::ZERO;

        assert_eq!(y, x);
    }

    #[test]
    fn addition_is_correct() {
        let (x, y) = get_2_test_polynomial();

        let field_cfg = test_config();

        let res = DynamicPolynomialF::new(vec![
            F::from_with_cfg(3i32, &field_cfg),
            F::from_with_cfg(2i32, &field_cfg),
            F::from_with_cfg(5i32, &field_cfg),
            F::zero_with_cfg(&field_cfg),
            F::zero_with_cfg(&field_cfg),
        ]);

        assert_eq!(x.clone() + &y, res);
        assert_eq!(y.clone() + &x, res);
        assert_eq!(x.clone() + y.clone(), res);
        assert_eq!(x.checked_add(&y), Some(res.clone()));

        let mut z = x.clone();
        z += y.clone();
        assert_eq!(z, res);

        let mut z = x.clone();
        z += &y;
        assert_eq!(z, res);

        let mut z = y.clone();
        z += x.clone();
        assert_eq!(z, res);

        let mut z = y.clone();
        z += &x;
        assert_eq!(z, res);
    }

    #[test]
    fn subtraction_is_correct() {
        let (x, y) = get_2_test_polynomial();

        let field_cfg = test_config();

        let res = DynamicPolynomialF::new(vec![
            F::from_with_cfg(1i32, &field_cfg),
            F::from_with_cfg(-2i32, &field_cfg),
            F::from_with_cfg(-1i32, &field_cfg),
            F::zero_with_cfg(&field_cfg),
            F::zero_with_cfg(&field_cfg),
        ]);

        assert_eq!(x.clone() - &y, res);
        assert_eq!(x.clone() - y.clone(), res);
        assert_eq!(x.checked_sub(&y), Some(res.clone()));

        let mut z = x.clone();
        z -= y.clone();
        assert_eq!(z, res);

        let mut z = x.clone();
        z -= &y;
        assert_eq!(z, res);

        let x = y;
        let y = DynamicPolynomialF::new(vec![
            F::from_with_cfg(2i32, &field_cfg),
            F::from_with_cfg(0i32, &field_cfg),
            F::from_with_cfg(2i32, &field_cfg),
            F::from_with_cfg(-1i32, &field_cfg),
            F::zero_with_cfg(&field_cfg),
        ]);

        let res = DynamicPolynomialF::new(vec![
            F::from_with_cfg(-1i32, &field_cfg),
            F::from_with_cfg(2i32, &field_cfg),
            F::from_with_cfg(1i32, &field_cfg),
            F::from_with_cfg(1i32, &field_cfg),
            F::zero_with_cfg(&field_cfg),
        ]);

        assert_eq!(x.clone() - &y, res);
        assert_eq!(x.clone() - y.clone(), res);
        assert_eq!(x.checked_sub(&y), Some(res.clone()));

        let mut z = x.clone();
        z -= y.clone();
        assert_eq!(z, res);

        let mut z = x.clone();
        z -= &y;
        assert_eq!(z, res);
    }

    #[test]
    fn mul_zero() {
        assert_eq!(
            DynamicPolynomialF::<F>::ZERO * DynamicPolynomialF::ZERO,
            DynamicPolynomialF::ZERO
        );

        let field_cfg = test_config();

        let x = DynamicPolynomialF::new(vec![
            F::from_with_cfg(2i32, &field_cfg),
            F::zero_with_cfg(&field_cfg),
            F::from_with_cfg(2i32, &field_cfg),
            F::zero_with_cfg(&field_cfg),
            F::zero_with_cfg(&field_cfg),
        ]);

        assert_eq!(
            x.clone() * &DynamicPolynomialF::ZERO,
            DynamicPolynomialF::ZERO
        );
        assert_eq!(
            DynamicPolynomialF::ZERO * x.clone(),
            DynamicPolynomialF::ZERO
        );
        assert_eq!(DynamicPolynomialF::ZERO * &x, DynamicPolynomialF::ZERO);

        let mut y = x.clone();

        y *= DynamicPolynomialF::ZERO;

        assert_eq!(y, DynamicPolynomialF::ZERO);

        y *= &DynamicPolynomialF::ZERO;

        assert_eq!(y, DynamicPolynomialF::ZERO);
    }

    #[test]
    fn multiplication_is_correct() {
        let (x, y) = get_2_test_polynomial();

        let field_cfg = test_config();

        let res = DynamicPolynomialF::new(vec![
            F::from_with_cfg(2i32, &field_cfg),
            F::from_with_cfg(4i32, &field_cfg),
            F::from_with_cfg(8i32, &field_cfg),
            F::from_with_cfg(4i32, &field_cfg),
            F::from_with_cfg(6i32, &field_cfg),
            F::zero_with_cfg(&field_cfg),
            F::zero_with_cfg(&field_cfg),
        ]);

        assert_eq!(x.clone() * y.clone(), res);
        assert_eq!(x.clone() * &y, res);
        assert_eq!(&x * &y, res);
        assert_eq!(y.clone() * x.clone(), res);
        assert_eq!(y.clone() * &x, res);
        assert_eq!(&y * &x, res);
        assert_eq!(x.checked_mul(&y), Some(res.clone()));
        assert_eq!(y.checked_mul(&x), Some(res.clone()));

        let mut z = x.clone();
        z *= y.clone();
        assert_eq!(z, res);

        let mut z = x.clone();
        z *= &y;
        assert_eq!(z, res);

        let mut z = y.clone();
        z *= x.clone();
        assert_eq!(z, res);

        let mut z = y.clone();
        z *= &x;
        assert_eq!(z, res);
    }

    #[test]
    fn test_trim() {
        let field_cfg = test_config();

        let mut x = DynamicPolynomialF::new([
            F::zero_with_cfg(&field_cfg),
            F::zero_with_cfg(&field_cfg),
            F::zero_with_cfg(&field_cfg),
            F::zero_with_cfg(&field_cfg),
            F::zero_with_cfg(&field_cfg),
        ]);
        x.trim();
        assert_eq!(x, DynamicPolynomialF::ZERO);

        let mut x = DynamicPolynomialF::new([
            F::from_with_cfg(2i32, &field_cfg),
            F::from_with_cfg(3i32, &field_cfg),
            F::zero_with_cfg(&field_cfg),
            F::zero_with_cfg(&field_cfg),
            F::zero_with_cfg(&field_cfg),
        ]);
        x.trim();
        assert_eq!(
            x,
            DynamicPolynomialF::new([
                F::from_with_cfg(2i32, &field_cfg),
                F::from_with_cfg(3i32, &field_cfg),
            ])
        );
    }

    #[test]
    fn evaluate_zero_poly() {
        assert_eq!(
            DynamicPolynomialF::<F>::ZERO.evaluate_at_point(&F::one_with_cfg(&test_config())),
            Ok(F::zero_with_cfg(&test_config()))
        )
    }

    fn f(v: u64) -> F {
        F::from_with_cfg(v, &test_config())
    }

    #[test]
    #[should_panic(expected = "division by zero polynomial")]
    fn test_div_rem_zero_divisor() {
        let dividend = DynamicPolynomialF {
            coeffs: vec![f(1), f(1)],
        };
        let divisor = DynamicPolynomialF {
            coeffs: vec![f(0), f(0)],
        };

        let _ = dividend.div_rem_euclid(&divisor);
    }

    #[test]
    fn test_div_euclid_and_rem_euclid() {
        let dividend = DynamicPolynomialF {
            coeffs: vec![f(1), f(2), f(1)],
        };
        let divisor = DynamicPolynomialF {
            coeffs: vec![f(1), f(1)],
        };

        let (q, r) = dividend.div_rem_euclid(&divisor);
        let q2 = dividend.div_euclid(&divisor);
        let r2 = dividend.rem_euclid(&divisor);

        assert_eq!(q, q2);
        assert_eq!(r, r2);
    }

    fn any_f() -> impl Strategy<Value = F> {
        let cfg = test_config();
        any::<u64>().prop_map(move |v| F::from_with_cfg(v, &cfg))
    }

    fn any_poly(max_degree: usize) -> impl Strategy<Value = DynamicPolynomialF<F>> {
        let cfg = test_config();
        (0usize..=max_degree).prop_flat_map(move |degree| {
            let cfg = cfg;
            prop::collection::vec(any_f(), degree + 1).prop_map(move |mut coeffs| {
                let mut rng = rand::rngs::ThreadRng::default();
                let zero = F::from_with_cfg(0, &cfg);
                while coeffs[degree] == zero {
                    let val: u64 = rng.random();
                    coeffs[degree] = F::from_with_cfg(val, &cfg);
                }
                DynamicPolynomialF { coeffs }
            })
        })
    }

    fn sparse_poly(max_degree: usize) -> impl Strategy<Value = DynamicPolynomialF<F>> {
        let cfg = test_config();
        (1usize..=max_degree).prop_flat_map(move |degree| {
            let cfg = cfg;
            let num_nonzero = std::cmp::min(3, degree);
            (
                prop::collection::vec(0usize..degree, num_nonzero),
                prop::collection::vec(any_f(), num_nonzero + 1),
            )
                .prop_map(move |(positions, values)| {
                    let mut rng = rand::rngs::ThreadRng::default();
                    let zero = F::from_with_cfg(0, &cfg);
                    let mut coeffs = vec![zero.clone(); degree + 1];
                    positions
                        .iter()
                        .zip(values.iter())
                        .for_each(|(p, v)| coeffs[*p] = v.clone());
                    coeffs[degree] = values.last().unwrap().clone();
                    while coeffs[degree] == zero {
                        let val: u64 = rng.random();
                        coeffs[degree] = F::from_with_cfg(val, &cfg);
                    }
                    DynamicPolynomialF { coeffs }
                })
        })
    }

    fn small_coeff_poly(max_degree: usize) -> impl Strategy<Value = DynamicPolynomialF<F>> {
        let cfg = test_config();
        (0usize..=max_degree).prop_flat_map(move |degree| {
            let cfg = cfg;
            prop::collection::vec(0u64..=2, degree + 1).prop_map(move |raw| {
                let mut rng = rand::rngs::ThreadRng::default();
                let zero = F::from_with_cfg(0, &cfg);
                let mut coeffs: Vec<F> = raw.iter().map(|&v| F::from_with_cfg(v, &cfg)).collect();
                while coeffs[degree] == zero {
                    let val: u64 = rng.random::<u64>() % 2 + 1;
                    coeffs[degree] = F::from_with_cfg(val, &cfg);
                }
                DynamicPolynomialF { coeffs }
            })
        })
    }

    fn any_poly_varied(max_degree: usize) -> impl Strategy<Value = DynamicPolynomialF<F>> {
        prop_oneof![
            2 => any_poly(max_degree),
            1 => sparse_poly(max_degree),
            1 => small_coeff_poly(max_degree),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(500))]

        #[test]
        fn prop_div_rem_random(
            dividend in any_poly_varied(50),
            divisor in any_poly_varied(50)
        ) {
            let (quotient, remainder) = dividend.div_rem_euclid(&divisor);

            let product = &quotient * &divisor;
            let reconstructed = product + &remainder;

            let mut dividend_trimmed = dividend.clone();
            let mut reconstructed_trimmed = reconstructed;
            dividend_trimmed.trim();
            reconstructed_trimmed.trim();

            prop_assert_eq!(dividend_trimmed, reconstructed_trimmed);
        }

        #[test]
        fn prop_div_rem_exact_division(
            q in any_poly_varied(25),
            d in any_poly_varied(25)
        ) {
            let dividend = &q * &d;
            let (quotient, remainder) = dividend.div_rem_euclid(&d);

            let mut remainder_trimmed = remainder;
            let mut quotient_trimmed = quotient;
            remainder_trimmed.trim();
            quotient_trimmed.trim();

            let mut q_trimmed = q.clone();
            q_trimmed.trim();

            prop_assert_eq!(remainder_trimmed, DynamicPolynomialF::ZERO);
            prop_assert_eq!(quotient_trimmed, q_trimmed);
        }
    }
}
