use crypto_bigint::{BoxedUint, NonZero, Uint as CBUint};
use crypto_primitives::{
    Field, IntRing, PrimeField, crypto_bigint_boxed_monty::BoxedMontyField,
    crypto_bigint_const_monty::ConstMontyField, crypto_bigint_int::Int,
    crypto_bigint_monty::MontyField, crypto_bigint_uint::Uint,
};

use crate::inner_product::InnerProductError;

pub struct PreparedMontgomeryRhs<F: PrimeField> {
    // Values are shifted by one extra Montgomery factor so that multiplying by
    // lhs coefficients injected with `new_unchecked_with_cfg` produces an
    // ordinary field product.
    shifted_values: Vec<F>,
    cfg: F::Config,
}

/// Computes the same field result as `<lhs, rhs>` after mapping each signed
/// integer coefficient into the field, while avoiding full Montgomery
/// conversion for every lhs entry.
pub trait MontgomeryIntegerInnerProduct<Lhs>: PrimeField {
    type PreparedRhs: Sync;

    fn prepare_montgomery_rhs(rhs: &[Self], zero: &Self) -> Self::PreparedRhs;

    fn inner_product_prepared_montgomery(
        lhs: &[Lhs],
        rhs: &Self::PreparedRhs,
    ) -> Result<Self, InnerProductError>;

    fn inner_product_montgomery(
        lhs: &[Lhs],
        rhs: &[Self],
        zero: Self,
    ) -> Result<Self, InnerProductError> {
        let prepared = Self::prepare_montgomery_rhs(rhs, &zero);
        Self::inner_product_prepared_montgomery(lhs, &prepared)
    }
}

fn abs_as_field_width<const FIELD_LIMBS: usize, const INT_LIMBS: usize>(
    value: &Int<INT_LIMBS>,
    modulus: &NonZero<CBUint<FIELD_LIMBS>>,
) -> CBUint<FIELD_LIMBS> {
    let abs = value.inner().abs();
    if FIELD_LIMBS < INT_LIMBS {
        if abs.as_words()[FIELD_LIMBS..].iter().all(|word| *word == 0) {
            let abs = abs.resize();
            if abs < *modulus.as_ref() {
                return abs;
            }
        }
        let wide_modulus = NonZero::new(modulus.as_ref().resize::<INT_LIMBS>()).unwrap();
        abs.rem(&wide_modulus).resize()
    } else {
        let abs = abs.resize();
        if abs >= *modulus.as_ref() {
            abs.rem(modulus)
        } else {
            abs
        }
    }
}

impl<const FIELD_LIMBS: usize, const INT_LIMBS: usize> MontgomeryIntegerInnerProduct<Int<INT_LIMBS>>
    for MontyField<FIELD_LIMBS>
{
    type PreparedRhs = PreparedMontgomeryRhs<Self>;

    fn prepare_montgomery_rhs(rhs: &[Self], zero: &Self) -> Self::PreparedRhs {
        let cfg = *zero.cfg();
        let shifted_values = rhs
            .iter()
            .map(|q| {
                assert_eq!(q.cfg().modulus(), cfg.modulus());
                Self::new_with_cfg(*q.inner(), &cfg)
            })
            .collect();
        PreparedMontgomeryRhs {
            shifted_values,
            cfg,
        }
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn inner_product_prepared_montgomery(
        lhs: &[Int<INT_LIMBS>],
        rhs: &Self::PreparedRhs,
    ) -> Result<Self, InnerProductError> {
        if lhs.len() != rhs.shifted_values.len() {
            return Err(InnerProductError::LengthMismatch {
                lhs: lhs.len(),
                rhs: rhs.shifted_values.len(),
            });
        }

        let cfg = rhs.cfg;
        let modulus = cfg.modulus().as_nz_ref();
        let mut acc = Self::zero_with_cfg(&cfg);

        for (coeff, q) in lhs.iter().zip(&rhs.shifted_values) {
            let abs = abs_as_field_width(coeff, modulus);
            let term = Self::new_unchecked_with_cfg(Uint::new(abs), &cfg) * q;
            if coeff.is_negative() {
                acc -= &term;
            } else {
                acc += &term;
            }
        }

        Ok(acc)
    }
}

impl<
    Mod: crypto_bigint::modular::ConstMontyParams<FIELD_LIMBS>,
    const FIELD_LIMBS: usize,
    const INT_LIMBS: usize,
> MontgomeryIntegerInnerProduct<Int<INT_LIMBS>> for ConstMontyField<Mod, FIELD_LIMBS>
{
    type PreparedRhs = PreparedMontgomeryRhs<Self>;

    fn prepare_montgomery_rhs(rhs: &[Self], _zero: &Self) -> Self::PreparedRhs {
        let shifted_values = rhs
            .iter()
            .map(|q| Self::new_with_cfg(*q.inner(), &()))
            .collect();
        PreparedMontgomeryRhs {
            shifted_values,
            cfg: (),
        }
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn inner_product_prepared_montgomery(
        lhs: &[Int<INT_LIMBS>],
        rhs: &Self::PreparedRhs,
    ) -> Result<Self, InnerProductError> {
        if lhs.len() != rhs.shifted_values.len() {
            return Err(InnerProductError::LengthMismatch {
                lhs: lhs.len(),
                rhs: rhs.shifted_values.len(),
            });
        }

        let modulus = Mod::PARAMS.modulus().as_nz_ref();
        let mut acc = Self::zero_with_cfg(&());

        for (coeff, q) in lhs.iter().zip(&rhs.shifted_values) {
            let abs = abs_as_field_width(coeff, modulus);
            let term = Self::new_unchecked_with_cfg(Uint::new(abs), &()) * q;
            if coeff.is_negative() {
                acc -= &term;
            } else {
                acc += &term;
            }
        }

        Ok(acc)
    }
}

impl<const INT_LIMBS: usize> MontgomeryIntegerInnerProduct<Int<INT_LIMBS>> for BoxedMontyField {
    type PreparedRhs = PreparedMontgomeryRhs<Self>;

    fn prepare_montgomery_rhs(rhs: &[Self], zero: &Self) -> Self::PreparedRhs {
        let cfg = zero.cfg().clone();
        let shifted_values = rhs
            .iter()
            .map(|q| {
                assert_eq!(q.cfg().modulus(), cfg.modulus());
                Self::new_with_cfg(q.inner().clone(), &cfg)
            })
            .collect();
        PreparedMontgomeryRhs {
            shifted_values,
            cfg,
        }
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn inner_product_prepared_montgomery(
        lhs: &[Int<INT_LIMBS>],
        rhs: &Self::PreparedRhs,
    ) -> Result<Self, InnerProductError> {
        if lhs.len() != rhs.shifted_values.len() {
            return Err(InnerProductError::LengthMismatch {
                lhs: lhs.len(),
                rhs: rhs.shifted_values.len(),
            });
        }

        let cfg = rhs.cfg.clone();
        let modulus = cfg.modulus().as_nz_ref();
        let mut acc = Self::zero_with_cfg(&cfg);

        for (coeff, q) in lhs.iter().zip(&rhs.shifted_values) {
            let abs: BoxedUint = coeff.inner().abs().into();
            let abs = abs.rem(modulus);
            let term = Self::new_unchecked_with_cfg(abs, &cfg) * q;
            if coeff.is_negative() {
                acc -= &term;
            } else {
                acc += &term;
            }
        }

        Ok(acc)
    }
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects)]
mod tests {
    use super::*;
    use crypto_bigint::{BoxedUint, U64, Word, const_monty_params};
    use crypto_primitives::{FromWithConfig, crypto_bigint_boxed_monty::BoxedMontyField};
    use proptest::prelude::*;

    use crate::inner_product::MBSInnerProduct;

    const FIELD_LIMBS: usize = U64::LIMBS;
    const INT_LIMBS: usize = U64::LIMBS * 2;
    const_monty_params!(Params7, U64, "0000000000000007");

    type FixedF = MontyField<FIELD_LIMBS>;
    type BoxedF = BoxedMontyField;
    type ConstF = ConstMontyField<Params7, FIELD_LIMBS>;

    fn large_positive() -> Int<INT_LIMBS> {
        Int::from_words([0, 1_u64.wrapping_shl(16) as Word])
    }

    fn fixed_cfg() -> <FixedF as PrimeField>::Config {
        FixedF::make_cfg(&Uint::new(CBUint::from(7_u8))).expect("odd modulus")
    }

    fn boxed_cfg() -> <BoxedF as PrimeField>::Config {
        BoxedF::make_cfg(&BoxedUint::from(7_u8)).expect("odd modulus")
    }

    fn fixed_cfg_11() -> <FixedF as PrimeField>::Config {
        FixedF::make_cfg(&Uint::new(CBUint::from(11_u8))).expect("odd modulus")
    }

    fn fixed_high_word_cfg() -> <FixedF as PrimeField>::Config {
        FixedF::make_cfg(&Uint::new(CBUint::from_words([0xffff_ffff_ffff_ffc5])))
            .expect("odd modulus")
    }

    fn boxed_cfg_11() -> <BoxedF as PrimeField>::Config {
        BoxedF::make_cfg(&BoxedUint::from(11_u8)).expect("odd modulus")
    }

    fn boxed_wide_cfg() -> <BoxedF as PrimeField>::Config {
        BoxedF::make_cfg(&BoxedUint::from((1_u128 << 127) - 1)).expect("odd modulus")
    }

    fn coeffs() -> [Int<INT_LIMBS>; 3] {
        [large_positive(), Int::from(-5_i32), Int::from(11_i32)]
    }

    fn dense_coeffs() -> Vec<Int<INT_LIMBS>> {
        (0_u64..64)
            .map(|i| {
                let mut words = [0; INT_LIMBS];
                words[0] = i * 37 + 5;
                words[1] = if i % 5 == 0 { 1 << 20 } else { 0 };
                let value = Int::from_words(words);
                if i % 2 == 0 { -value } else { value }
            })
            .collect()
    }

    fn rhs_values(len: usize) -> Vec<u64> {
        (0..len).map(|i| (i as u64 * 11 + 2) % 7).collect()
    }

    fn signed_int(lo: Word, hi: Word, is_negative: bool) -> Int<INT_LIMBS> {
        let value = Int::from_words([lo, hi]);
        if is_negative { -value } else { value }
    }

    fn manual_mod7(coeffs: &[Int<INT_LIMBS>], rhs: &[u64]) -> u64 {
        let modulus = NonZero::new(CBUint::<INT_LIMBS>::from(7_u8)).unwrap();
        let mut acc = 0_i128;

        for (coeff, rhs) in coeffs.iter().zip(rhs) {
            let abs_mod = i128::from(coeff.inner().abs().rem(&modulus).as_words()[0]);
            let term = (abs_mod * i128::from(*rhs)) % 7;
            if coeff.is_negative() {
                acc -= term;
            } else {
                acc += term;
            }
        }

        u64::try_from(acc.rem_euclid(7)).unwrap()
    }

    #[test]
    fn fixed_montgomery_inner_product_matches_manual_modular_sum() {
        let cfg = fixed_cfg();
        let rhs = [2_u64, 3, 4].map(|x| FixedF::from_with_cfg(x, &cfg));
        let expected = FixedF::from_with_cfg(2_u64, &cfg);

        let actual =
            FixedF::inner_product_montgomery(&coeffs(), &rhs, FixedF::zero_with_cfg(&cfg)).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn fixed_prepared_inner_product_matches_field_lift_for_dense_signed_coefficients() {
        let cfg = fixed_cfg();
        let coeffs = dense_coeffs();
        let rhs_values = rhs_values(coeffs.len());
        let rhs: Vec<_> = rhs_values
            .iter()
            .map(|x| FixedF::from_with_cfg(*x, &cfg))
            .collect();
        let zero = FixedF::zero_with_cfg(&cfg);
        let expected =
            MBSInnerProduct::inner_product_field::<_, FixedF>(&coeffs, &rhs, zero.clone()).unwrap();
        let prepared =
            <FixedF as MontgomeryIntegerInnerProduct<Int<INT_LIMBS>>>::prepare_montgomery_rhs(
                &rhs, &zero,
            );

        let actual = FixedF::inner_product_prepared_montgomery(&coeffs, &prepared).unwrap();

        assert_eq!(actual, expected);
        assert_eq!(
            actual,
            FixedF::from_with_cfg(manual_mod7(&coeffs, &rhs_values), &cfg)
        );
    }

    #[test]
    fn fixed_prepared_inner_product_matches_field_lift_for_high_word_modulus() {
        let cfg = fixed_high_word_cfg();
        let zero = FixedF::zero_with_cfg(&cfg);
        let coeffs = [
            Int::from_words([0xffff_ffff_ffff_ff80, 0]),
            -Int::from_words([0xdead_beef_cafe_babe, 1]),
            Int::from_words([0x8000_0000_0000_0041, 2]),
            -Int::from_words([0x1234_5678_9abc_def0, 0]),
        ];
        let rhs_values: [Word; 4] = [
            0xffff_ffff_ffff_ffc4,
            0xdead_beef_cafe_babe,
            0x8000_0000_0000_0041,
            0x1234_5678_9abc_def0,
        ];
        let rhs =
            rhs_values.map(|value| FixedF::new_with_cfg(Uint::new(CBUint::from(value)), &cfg));
        let expected =
            MBSInnerProduct::inner_product_field::<_, FixedF>(&coeffs, &rhs, zero.clone()).unwrap();
        let prepared =
            <FixedF as MontgomeryIntegerInnerProduct<Int<INT_LIMBS>>>::prepare_montgomery_rhs(
                &rhs, &zero,
            );

        let actual = FixedF::inner_product_prepared_montgomery(&coeffs, &prepared).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn boxed_montgomery_inner_product_handles_wide_coefficients() {
        let cfg = boxed_cfg();
        let rhs = [2_u64, 3, 4].map(|x| BoxedF::from_with_cfg(x, &cfg));
        let expected = BoxedF::from_with_cfg(2_u64, &cfg);

        let actual =
            BoxedF::inner_product_montgomery(&coeffs(), &rhs, BoxedF::zero_with_cfg(&cfg)).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn boxed_prepared_inner_product_matches_manual_modular_sum_for_dense_signed_coefficients() {
        let cfg = boxed_cfg();
        let coeffs = dense_coeffs();
        let rhs_values = rhs_values(coeffs.len());
        let rhs: Vec<_> = rhs_values
            .iter()
            .map(|x| BoxedF::from_with_cfg(*x, &cfg))
            .collect();
        let zero = BoxedF::zero_with_cfg(&cfg);
        let expected = BoxedF::from_with_cfg(manual_mod7(&coeffs, &rhs_values), &cfg);
        let prepared =
            <BoxedF as MontgomeryIntegerInnerProduct<Int<INT_LIMBS>>>::prepare_montgomery_rhs(
                &rhs, &zero,
            );

        let actual = BoxedF::inner_product_prepared_montgomery(&coeffs, &prepared).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn const_montgomery_inner_product_matches_manual_modular_sum() {
        let rhs = [2_u64, 3, 4].map(ConstF::from);
        let expected = ConstF::from(2_u64);

        let actual =
            ConstF::inner_product_montgomery(&coeffs(), &rhs, ConstF::zero_with_cfg(&())).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn const_prepared_inner_product_matches_manual_modular_sum_for_dense_signed_coefficients() {
        let coeffs = dense_coeffs();
        let rhs_values = rhs_values(coeffs.len());
        let rhs: Vec<_> = rhs_values.iter().copied().map(ConstF::from).collect();
        let expected = ConstF::from(manual_mod7(&coeffs, &rhs_values));
        let prepared =
            <ConstF as MontgomeryIntegerInnerProduct<Int<INT_LIMBS>>>::prepare_montgomery_rhs(
                &rhs,
                &ConstF::zero_with_cfg(&()),
            );

        let actual = ConstF::inner_product_prepared_montgomery(&coeffs, &prepared).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn montgomery_inner_product_rejects_length_mismatch() {
        let cfg = fixed_cfg();
        let err = FixedF::inner_product_montgomery(
            &[Int::<INT_LIMBS>::from(1_i32)],
            &[],
            FixedF::zero_with_cfg(&cfg),
        )
        .unwrap_err();

        assert_eq!(err, InnerProductError::LengthMismatch { lhs: 1, rhs: 0 });
    }

    #[test]
    #[should_panic]
    fn fixed_prepare_rejects_mismatched_dynamic_field_config() {
        let cfg = fixed_cfg();
        let other_cfg = fixed_cfg_11();
        let rhs = [FixedF::from_with_cfg(1_u64, &other_cfg)];

        <FixedF as MontgomeryIntegerInnerProduct<Int<INT_LIMBS>>>::prepare_montgomery_rhs(
            &rhs,
            &FixedF::zero_with_cfg(&cfg),
        );
    }

    #[test]
    fn fixed_prepared_inner_product_uses_prepared_config() {
        let other_cfg = fixed_cfg_11();
        let rhs = [FixedF::from_with_cfg(1_u64, &other_cfg)];
        let prepared =
            <FixedF as MontgomeryIntegerInnerProduct<Int<INT_LIMBS>>>::prepare_montgomery_rhs(
                &rhs,
                &FixedF::zero_with_cfg(&other_cfg),
            );

        let actual =
            FixedF::inner_product_prepared_montgomery(&[Int::<INT_LIMBS>::from(1_i32)], &prepared)
                .unwrap();

        assert_eq!(actual, FixedF::from_with_cfg(1_u64, &other_cfg));
    }

    #[test]
    #[should_panic]
    fn boxed_prepare_rejects_mismatched_dynamic_field_config() {
        let cfg = boxed_cfg();
        let other_cfg = boxed_cfg_11();
        let rhs = [BoxedF::from_with_cfg(1_u64, &other_cfg)];

        <BoxedF as MontgomeryIntegerInnerProduct<Int<INT_LIMBS>>>::prepare_montgomery_rhs(
            &rhs,
            &BoxedF::zero_with_cfg(&cfg),
        );
    }

    #[test]
    fn boxed_prepared_inner_product_uses_prepared_config() {
        let other_cfg = boxed_cfg_11();
        let rhs = [BoxedF::from_with_cfg(1_u64, &other_cfg)];
        let prepared =
            <BoxedF as MontgomeryIntegerInnerProduct<Int<INT_LIMBS>>>::prepare_montgomery_rhs(
                &rhs,
                &BoxedF::zero_with_cfg(&other_cfg),
            );

        let actual =
            BoxedF::inner_product_prepared_montgomery(&[Int::<INT_LIMBS>::from(1_i32)], &prepared)
                .unwrap();

        assert_eq!(actual, BoxedF::from_with_cfg(1_u64, &other_cfg));
    }

    proptest! {
        #[test]
        #[cfg_attr(miri, ignore)]
        fn fixed_prepared_inner_product_matches_field_lift_for_random_dense_inputs(
            entries in prop::collection::vec(
                (any::<Word>(), 0_u64..(1_u64 << 20), any::<bool>(), any::<Word>()),
                0..64,
            )
        ) {
            let cfg = fixed_high_word_cfg();
            let zero = FixedF::zero_with_cfg(&cfg);
            let coeffs: Vec<_> = entries
                .iter()
                .map(|(lo, hi, is_negative, _)| signed_int(*lo, *hi, *is_negative))
                .collect();
            let rhs: Vec<_> = entries
                .iter()
                .map(|(_, _, _, rhs)| FixedF::new_with_cfg(Uint::new(CBUint::from(*rhs)), &cfg))
                .collect();
            let expected =
                MBSInnerProduct::inner_product_field::<_, FixedF>(&coeffs, &rhs, zero.clone())
                    .unwrap();
            let prepared =
                <FixedF as MontgomeryIntegerInnerProduct<Int<INT_LIMBS>>>::prepare_montgomery_rhs(
                    &rhs,
                    &zero,
                );

            let actual = FixedF::inner_product_prepared_montgomery(&coeffs, &prepared).unwrap();

            prop_assert_eq!(actual, expected);
        }

        #[test]
        #[cfg_attr(miri, ignore)]
        fn boxed_prepared_inner_product_matches_field_lift_for_random_dense_inputs(
            entries in prop::collection::vec(
                (any::<Word>(), 0_u64..(1_u64 << 20), any::<bool>(), any::<Word>()),
                0..64,
            )
        ) {
            let cfg = boxed_wide_cfg();
            let zero = BoxedF::zero_with_cfg(&cfg);
            let coeffs: Vec<_> = entries
                .iter()
                .map(|(lo, hi, is_negative, _)| signed_int(*lo, *hi, *is_negative))
                .collect();
            let rhs: Vec<_> = entries
                .iter()
                .map(|(_, _, _, rhs)| BoxedF::from_with_cfg(*rhs, &cfg))
                .collect();
            let expected =
                MBSInnerProduct::inner_product_field::<_, BoxedF>(&coeffs, &rhs, zero.clone())
                    .unwrap();
            let prepared =
                <BoxedF as MontgomeryIntegerInnerProduct<Int<INT_LIMBS>>>::prepare_montgomery_rhs(
                    &rhs,
                    &zero,
                );

            let actual = BoxedF::inner_product_prepared_montgomery(&coeffs, &prepared).unwrap();

            prop_assert_eq!(actual, expected);
        }
    }
}
