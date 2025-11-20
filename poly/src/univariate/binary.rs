//! Dense polynomial with binary coefficients.
use std::{fmt::Debug, ops::BitAnd};

use crypto_primitives::PrimeField;
use num_traits::ConstZero;
use zinc_utils::projectable_to_field::ProjectableToField;

use crate::{EvaluatablePolynomial, EvaluationError, Polynomial};

// A type used to store the coefficients of
// the binary polynomials. In our case, it is
// either `u32` or `u64`.
pub trait BinaryPolyCarrier:
    ConstZero + TryFrom<u64, Error: Debug> + for<'a> BitAnd<&'a Self, Output = Self> + Sized + Eq
{
    /// The bit size of the type.
    const BIT_SIZE: u32;
}

impl BinaryPolyCarrier for u32 {
    const BIT_SIZE: u32 = 32;
}

impl BinaryPolyCarrier for u64 {
    const BIT_SIZE: u32 = 64;
}

pub struct BinaryPoly<T> {
    pub coeffs: T,
}

impl<T: BinaryPolyCarrier> Polynomial<bool> for BinaryPoly<T> {
    const DEGREE_BOUND: usize = T::BIT_SIZE as usize;
}

impl<T: BinaryPolyCarrier> BinaryPoly<T> {
    /// Create a polynomial of the form `X^pow_of_x`.
    /// If `pow_of_x` exceeds the maximum degree of polynomials
    /// of this type `None` is returned.
    #[inline(always)]
    pub fn single_term(pow_of_x: u32) -> Option<Self> {
        if pow_of_x >= T::BIT_SIZE {
            return None;
        }

        let coeffs = (1 << pow_of_x)
            .try_into()
            .expect("1 << pow_of_x is always less than the max T, since T::BIT_SIZE < pow_of_x");
        Some(Self { coeffs })
    }
}

impl<T: BinaryPolyCarrier> From<T> for BinaryPoly<T> {
    #[inline(always)]
    fn from(x: T) -> Self {
        Self { coeffs: x }
    }
}

impl<T: BinaryPolyCarrier, F: PrimeField> EvaluatablePolynomial<bool, F, F> for BinaryPoly<T> {
    type EvaluationPoint = F;

    #[allow(clippy::arithmetic_side_effects)] // (T::BIT_SIZE - 1) - i is fine if 0 <= i < T::BIT_SIZE.
    fn evaluate_at_point(&self, point: &Self::EvaluationPoint) -> Result<F, EvaluationError> {
        // Horner's method.
        let one = F::one_with_cfg(point.cfg());
        Ok((0..T::BIT_SIZE)
            .map(|i| {
                T::try_from(1u64 << ((T::BIT_SIZE - 1) - i))
                    .expect("T is either u32 or u64 and this should always go through.")
                    & &self.coeffs
            })
            .fold(F::zero_with_cfg(point.cfg()), |mut acc, coeff| {
                acc *= point;
                if coeff != T::ZERO { acc + &one } else { acc }
            }))
    }
}

impl<T: BinaryPolyCarrier, F: PrimeField + 'static> ProjectableToField<F> for BinaryPoly<T> {
    #[allow(clippy::arithmetic_side_effects)]
    fn prepare_projection(sampled_value: &F) -> impl Fn(&Self) -> F + 'static {
        let field_cfg = sampled_value.cfg().clone();
        let r_powers = {
            // It makes sense to preprocess the powers here
            // since the evaluation procedure
            // will reduce to additions only in that case.
            let mut r_powers = Vec::with_capacity(T::BIT_SIZE as usize);

            let mut curr = F::one_with_cfg(&field_cfg);
            r_powers.push(curr.clone());

            for _ in 1..32 {
                curr *= sampled_value;
                r_powers.push(curr.clone());
            }

            r_powers
        };

        move |Self { coeffs, .. }| {
            (0..T::BIT_SIZE)
                .filter(|i| {
                    !(T::try_from(1 << i)
                        .expect("If T is u32 i is 31 at most. If T is u64 i is 63 at most.")
                        & coeffs)
                        .is_zero()
                })
                .fold(F::zero_with_cfg(&field_cfg), |acc, i| {
                    acc + &r_powers[i as usize]
                })
        }
    }
}

#[cfg(test)]
mod test {

    use crypto_bigint::{Odd, U128, const_monty_params, modular::MontyParams};

    use crypto_primitives::{
        FromWithConfig, crypto_bigint_const_monty::ConstMontyField, crypto_bigint_monty::F256,
    };
    use itertools::Itertools;
    use zinc_utils::projectable_to_field::ProjectableToField;

    use crate::{EvaluatablePolynomial, univariate::binary::BinaryPoly};

    const N: usize = 2;

    const_monty_params!(Params, U128, "00000000b933426489189cb5b47d567f");

    type F = ConstMontyField<Params, N>;

    #[test]
    fn test_evaluate_binary_poly() {
        for i in 0..u32::from(u16::MAX) {
            // A u32 treated as a binary poly should give
            // its own value when evaluated on 2.
            let x = BinaryPoly::<u32>::from(i);

            let x_val_on_1 = x.evaluate_at_point(&F::from(2)).unwrap();

            assert_eq!(x_val_on_1, F::from(i));
        }
    }

    #[test]
    fn test_project_onto_field() {
        let project = BinaryPoly::<u32>::prepare_projection(&F::from(2));
        for i in 0..u32::from(u16::MAX) {
            assert_eq!(project(&BinaryPoly::from(i)), F::from(i));
        }
    }

    #[test]
    fn x_value_on_2() {
        for i in 0..64 {
            let x = BinaryPoly::<u64>::single_term(i).unwrap();
            let x_val = x.evaluate_at_point(&F::from(2)).unwrap();
            // 1u64 to not fall into i64.
            assert_eq!(x_val, F::from(1u64 << i));
        }
    }

    const LIMBS: usize = 4;
    type FMonty = F256;

    fn test_monty_config() -> MontyParams<LIMBS> {
        // Using a 256-bit prime 2^256 - 2^32 - 977 (secp256k1 field prime)
        let modulus = crypto_bigint::Uint::<LIMBS>::from_be_hex(
            "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f",
        );
        let modulus = Odd::new(modulus).expect("modulus should be odd");
        MontyParams::new(modulus)
    }

    #[test]
    fn test_projection_monty() {
        let x = FMonty::from_with_cfg(2, &test_monty_config());

        let v = (0..1024).map(BinaryPoly::<u32>::from).collect_vec();

        let project = BinaryPoly::<u32>::prepare_projection(&x);

        for (i, el) in v.iter().enumerate() {
            assert_eq!(
                project(el),
                FMonty::from_with_cfg(i as u64, &test_monty_config())
            );
        }
    }
}
