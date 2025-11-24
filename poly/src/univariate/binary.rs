//! Dense polynomial with binary coefficients.
use std::{fmt::Debug, ops::BitAnd};

use crypto_primitives::PrimeField;
use zinc_utils::projectable_to_field::ProjectableToField;

use crate::{EvaluatablePolynomial, EvaluationError, Polynomial};

// A type used to store the coefficients of
// the binary polynomials. In our case, it is
// either `u32` or `u64`.
pub trait BinaryPolyCarrier:
    TryFrom<u64, Error: Debug> + for<'a> BitAnd<&'a Self, Output = Self> + Sized + Eq
{
    /// The bit size of the type.
    const BIT_SIZE: u32;
    /// To avoid reallocations we might pad
    /// our binary polynomials. This type
    /// is for padding the carrier to 64 bits.
    /// It means for `u32` it must be `u32`,
    /// for `u64` - `()`.
    ///
    /// It is convenient to first pad
    /// the actual content of a polynomial
    /// to 64 bits and then to pad it to whatever
    /// size one needs.
    type PAD: Default;
}

impl BinaryPolyCarrier for u32 {
    const BIT_SIZE: u32 = 32;
    type PAD = u32;
}

impl BinaryPolyCarrier for u64 {
    const BIT_SIZE: u32 = 64;
    type PAD = ();
}

macro_rules! define_binary_poly {
    ($name:ident, $align:literal) => {
        pub struct $name<T: BinaryPolyCarrier> {
            pub coeffs: T,
            _pad_to_word: T::PAD,
            _padding: [u8; $align],
        }

        impl<T: BinaryPolyCarrier> Polynomial<bool> for $name<T> {
            const DEGREE_BOUND: usize = T::BIT_SIZE as usize;
        }

        impl<T: BinaryPolyCarrier> $name<T> {
            #[inline(always)]
            pub fn x(i: u32) -> Option<Self> {
                if i >= T::BIT_SIZE {
                    return None;
                }

                let coeffs = (1 << i).try_into().unwrap();
                Some(Self {
                    coeffs,
                    _pad_to_word: Default::default(),
                    _padding: [0; $align],
                })
            }
        }

        impl<T: BinaryPolyCarrier> From<T> for $name<T> {
            #[inline(always)]
            fn from(x: T) -> Self {
                Self {
                    coeffs: x,
                    _pad_to_word: Default::default(),
                    _padding: [0; $align],
                }
            }
        }

        impl<T: BinaryPolyCarrier, F: PrimeField> EvaluatablePolynomial<bool, F, F> for $name<T> {
            type EvaluationPoint = F;

            #[allow(clippy::arithmetic_side_effects)] // (T::BIT_SIZE - 1) - i is fine if 0 <= i < T::BIT_SIZE.
            fn evaluate_at_point(
                &self,
                point: &Self::EvaluationPoint,
            ) -> Result<F, EvaluationError> {
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
                        if coeff
                            != 0.try_into()
                                .expect("T is either u32 or u64 and 0u64 should be easily convertible into either :).")
                        {
                            acc + &one
                        } else {
                            acc
                        }
                    }))
            }
        }

        impl<T: BinaryPolyCarrier, F: PrimeField + 'static> ProjectableToField<F> for $name<T> {
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
                            T::try_from(1 << i)
                                .expect("If T is u32 i is 31 at most. If T is u64 i is 63 at most.")
                                & coeffs != T::try_from(0).expect("Zero should be fine, cf. above.")
                        })
                        .fold(F::zero_with_cfg(&field_cfg), |acc, i| {
                            acc + &r_powers[i as usize]
                        })
                }
            }
        }
    };
}

// The idea of these padded binary polynomials is
// to avoid reallocations.
// Say, we need to turn `Vec<BinaryPoly<u32>>`
// into `Vec<Uint<4>>` by evaluating each
// element of the vector on a point. If we had
// a straightforward representation of `BinaryPoly`
// we would have to allocate a new vector and then
// perform conversions. With padded `BinaryPoly` structures
// we can keep everything in place.

define_binary_poly!(BinaryPoly64, 0);
define_binary_poly!(BinaryPoly128, 8);
define_binary_poly!(BinaryPoly256, 24);

#[cfg(test)]
mod test {

    use crypto_bigint::{Odd, U128, const_monty_params, modular::MontyParams};

    use crypto_primitives::{
        FromWithConfig, crypto_bigint_const_monty::ConstMontyField, crypto_bigint_monty::F256,
    };
    use itertools::Itertools;
    use zinc_utils::projectable_to_field::ProjectableToField;

    use crate::{
        EvaluatablePolynomial,
        univariate::binary::{BinaryPoly64, BinaryPoly128, BinaryPoly256},
    };

    const N: usize = 2;

    const_monty_params!(Params, U128, "00000000b933426489189cb5b47d567f");

    type F = ConstMontyField<Params, N>;

    #[test]
    fn ensure_size() {
        assert_eq!(size_of::<BinaryPoly64<u32>>(), 8);
        assert_eq!(size_of::<BinaryPoly128<u32>>(), 16);
        assert_eq!(size_of::<BinaryPoly256<u32>>(), 32);
        assert_eq!(size_of::<BinaryPoly64<u64>>(), 8);
        assert_eq!(size_of::<BinaryPoly128<u64>>(), 16);
        assert_eq!(size_of::<BinaryPoly256<u64>>(), 32);
    }

    #[test]
    fn test_evaluate_binary_poly() {
        for i in 0..u32::from(u16::MAX) {
            // A u32 treated as a binary poly should give
            // its own value when evaluated on 2.
            let x = BinaryPoly128::<u32>::from(i);

            let x_val_on_1 = x.evaluate_at_point(&F::from(2)).unwrap();

            assert_eq!(x_val_on_1, F::from(i));
        }
    }

    #[test]
    fn test_project_onto_field() {
        let project = BinaryPoly128::<u32>::prepare_projection(&F::from(2));
        for i in 0..u32::from(u16::MAX) {
            assert_eq!(project(&BinaryPoly128::from(i)), F::from(i));
        }
    }

    #[test]
    fn x_value_on_2() {
        for i in 0..64 {
            let x = BinaryPoly256::<u64>::x(i).unwrap();
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

        let v = (0..1024).map(BinaryPoly256::<u32>::from).collect_vec();

        let project = BinaryPoly256::<u32>::prepare_projection(&x);

        for (i, el) in v.iter().enumerate() {
            assert_eq!(
                project(el),
                FMonty::from_with_cfg(i as u64, &test_monty_config())
            );
        }
    }
}
