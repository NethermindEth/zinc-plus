use crate::{UNCHECKED, from_ref::FromRef, mul_by_scalar::MulByScalar};
use crypto_bigint::{NonZero, Uint as CBUint};
use crypto_primitives::{
    BaseField, BaseFieldConfig, FixedConfig, ProjectElementWithConfig, SemiringConfig, Wrapper,
    boolean::Boolean,
    crypto_bigint_boxed_monty::BoxedMontyField,
    crypto_bigint_int::Int,
    crypto_bigint_monty::{MontyField, MontyFieldElement},
    crypto_bigint_uint::Uint,
};
use num_traits::{CheckedAdd, Signed};
use thiserror::Error;

/// A trait for inner product algorithms implementations.
///
/// `C` is the config needed to perform the operations: `()` for
/// self-sufficient types, or a field config for dynamic field elements.
pub trait InnerProduct<C, Lhs: ?Sized, Rhs, Output> {
    /// The main entry point for the inner product.
    /// `CHECK` determines whether the implementation should check for overflow.
    fn inner_product<const CHECK: bool>(
        cfg: &C,
        lhs: &Lhs,
        rhs: &[Rhs],
        zero: Output,
    ) -> Result<Output, InnerProductError>;
}

#[derive(Clone, Debug, PartialEq, Error)]
pub enum InnerProductError {
    #[error("The length of LHS and RHS does not match: LHS={lhs}, RHS={rhs}")]
    LengthMismatch { lhs: usize, rhs: usize },
    #[error("Arithmetic overflow")]
    Overflow,
}

/// An implementation of inner product that piggies back
/// on the `MulByScalar` and `CheckedAdd` traits.
/// It does `mul_by_scalar` for products of terms
/// and then combines the results using either `add` or `checked_add`.
#[derive(Clone, Debug)]
pub struct MBSInnerProduct;

impl<C, Lhs, Rhs, Out> InnerProduct<C, [Lhs], Rhs, Out> for MBSInnerProduct
where
    Out: FromRef<Lhs> + CheckedAdd + MulByScalar<Rhs, Out>,
{
    /// The mul-by-scalar inner product.
    #[allow(clippy::arithmetic_side_effects)] // Used in unchecked mode
    fn inner_product<const CHECK: bool>(
        _cfg: &C,
        lhs: &[Lhs],
        rhs: &[Rhs],
        zero: Out,
    ) -> Result<Out, InnerProductError> {
        if lhs.len() != rhs.len() {
            return Err(InnerProductError::LengthMismatch {
                lhs: lhs.len(),
                rhs: rhs.len(),
            });
        }

        lhs.iter().zip(rhs).try_fold(zero, |acc, (l, r)| {
            let widened = Out::from_ref(l);
            let product = widened
                .mul_by_scalar::<CHECK>(r)
                .ok_or(InnerProductError::Overflow)?;
            if CHECK {
                acc.checked_add(&product).ok_or(InnerProductError::Overflow)
            } else {
                Ok(acc + product)
            }
        })
    }
}

#[derive(Clone, Debug)]
pub struct NativeInnerProduct;

impl<C> InnerProduct<C, [C::Element], C::Element, C::Element> for NativeInnerProduct
where
    C: SemiringConfig,
{
    fn inner_product<const CHECK: bool>(
        cfg: &C,
        lhs: &[C::Element],
        rhs: &[C::Element],
        zero: C::Element,
    ) -> Result<C::Element, InnerProductError> {
        if lhs.len() != rhs.len() {
            return Err(InnerProductError::LengthMismatch {
                lhs: lhs.len(),
                rhs: rhs.len(),
            });
        }
        lhs.iter().zip(rhs).try_fold(zero, |mut acc, (l, r)| {
            let product = if CHECK {
                cfg.checked_mul(l, r).ok_or(InnerProductError::Overflow)?
            } else {
                cfg.mul(l, r)
            };
            if CHECK {
                acc = cfg
                    .checked_add(&acc, &product)
                    .ok_or(InnerProductError::Overflow)?;
            } else {
                cfg.add_assign(&mut acc, &product);
            }
            Ok(acc)
        })
    }
}

/// An implementation of inner product over a dynamic field: projects the RHS
/// entries into the field and folds with the field operations.
///
/// Field operations cannot overflow, so `CHECK` is ignored.
#[derive(Clone, Debug)]
pub struct FieldInnerProduct;

impl<C, Rhs> InnerProduct<C, [C::Element], Rhs, C::Element> for FieldInnerProduct
where
    C: SemiringConfig + ProjectElementWithConfig<Rhs>,
{
    fn inner_product<const CHECK: bool>(
        cfg: &C,
        lhs: &[C::Element],
        rhs: &[Rhs],
        zero: C::Element,
    ) -> Result<C::Element, InnerProductError> {
        if lhs.len() != rhs.len() {
            return Err(InnerProductError::LengthMismatch {
                lhs: lhs.len(),
                rhs: rhs.len(),
            });
        }

        Ok(lhs.iter().zip(rhs).fold(zero, |mut acc, (l, r)| {
            let product = cfg.mul(l, &cfg.project(r));
            cfg.add_assign(&mut acc, &product);
            acc
        }))
    }
}

/// Prepares a field-vector operand for repeated inner products with integer
/// rows.
///
/// Fixed-width runtime Montgomery fields override the ordinary projection
/// loop so a prepared field element replaces each per-coefficient Montgomery
/// conversion. Other supported field configurations retain the ordinary
/// projection semantics.
pub trait PreparedFieldInnerProduct<Rhs>: BaseFieldConfig + ProjectElementWithConfig<Rhs> {
    fn prepare_field_inner_product(
        &self,
        rhs: &[Self::Element],
    ) -> impl Fn(&[Rhs]) -> Result<Self::Element, InnerProductError> + Sync;
}

fn prepare_projecting_inner_product<C, Rhs>(
    cfg: &C,
    rhs: &[C::Element],
) -> impl Fn(&[Rhs]) -> Result<C::Element, InnerProductError> + Sync
where
    C: BaseFieldConfig + ProjectElementWithConfig<Rhs>,
{
    let cfg = cfg.clone();
    let rhs = rhs.to_vec();
    move |lhs| FieldInnerProduct::inner_product::<UNCHECKED>(&cfg, &rhs, lhs, cfg.zero())
}

impl<F, Rhs> PreparedFieldInnerProduct<Rhs> for FixedConfig<F>
where
    F: BaseField + for<'a> From<&'a Rhs>,
{
    fn prepare_field_inner_product(
        &self,
        rhs: &[Self::Element],
    ) -> impl Fn(&[Rhs]) -> Result<Self::Element, InnerProductError> + Sync {
        prepare_projecting_inner_product(self, rhs)
    }
}

impl<Rhs> PreparedFieldInnerProduct<Rhs> for BoxedMontyField
where
    BoxedMontyField: ProjectElementWithConfig<Rhs>,
{
    fn prepare_field_inner_product(
        &self,
        rhs: &[Self::Element],
    ) -> impl Fn(&[Rhs]) -> Result<Self::Element, InnerProductError> + Sync {
        prepare_projecting_inner_product(self, rhs)
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
        let wide_modulus =
            NonZero::<CBUint<INT_LIMBS>>::new_unwrap(modulus.as_ref().resize::<INT_LIMBS>());
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

impl<const FIELD_LIMBS: usize, const INT_LIMBS: usize> PreparedFieldInnerProduct<Int<INT_LIMBS>>
    for MontyField<FIELD_LIMBS>
{
    fn prepare_field_inner_product(
        &self,
        rhs: &[Self::Element],
    ) -> impl Fn(&[Int<INT_LIMBS>]) -> Result<Self::Element, InnerProductError> + Sync {
        let cfg = *self;
        // `q` is represented as qR. Projecting that raw representation once
        // produces qR², so multiplying it by an unshifted coefficient yields
        // the ordinary Montgomery result coefficient*qR.
        let shifted_rhs: Vec<_> = rhs.iter().map(|q| cfg.project(q.inner())).collect();

        move |lhs| {
            if lhs.len() != shifted_rhs.len() {
                return Err(InnerProductError::LengthMismatch {
                    lhs: lhs.len(),
                    rhs: shifted_rhs.len(),
                });
            }

            let modulus = cfg.params.modulus().as_nz_ref();
            let mut acc = cfg.zero();
            for (coeff, q) in lhs.iter().zip(&shifted_rhs) {
                let abs = abs_as_field_width(coeff, modulus);
                let raw_coeff = MontyFieldElement::from(Uint::new(abs));
                let term = cfg.mul(&raw_coeff, q);
                if coeff.is_negative() {
                    cfg.sub_assign(&mut acc, &term);
                } else {
                    cfg.add_assign(&mut acc, &term);
                }
            }
            Ok(acc)
        }
    }
}

/// The inner product for vectors of length 1 (a.k.a. scalars).
/// Uses `mul_by_scalar` to multiply the only components of vectors
/// to get the result.
#[derive(Clone, Debug)]
pub struct ScalarProduct;

impl<C, Lhs, Rhs, Out> InnerProduct<C, Lhs, Rhs, Out> for ScalarProduct
where
    Out: FromRef<Lhs> + MulByScalar<Rhs, Out>,
{
    /// A scalar inner product. Assumes `Lhs` is a scalar type
    /// and always asserts that `point` has only one component.
    fn inner_product<const CHECK: bool>(
        _cfg: &C,
        lhs: &Lhs,
        point: &[Rhs],
        _zero: Out,
    ) -> Result<Out, InnerProductError> {
        if point.as_ref().len() != 1 {
            Err(InnerProductError::LengthMismatch {
                lhs: 1,
                rhs: point.as_ref().len(),
            })
        } else {
            Ok(Out::from_ref(lhs)
                .mul_by_scalar::<CHECK>(&point[0])
                .ok_or(InnerProductError::Overflow)?)
        }
    }
}

/// The inner product for slices containing `Boolean` elements.
/// Uses `add` or `checked_add` to sum the elements of the RHS that
/// correspond to `true` elements of the boolean slice.
pub struct BooleanInnerProductAdd;

impl<C, Rhs: Clone, Out: FromRef<Rhs> + CheckedAdd> InnerProduct<C, [Boolean], Rhs, Out>
    for BooleanInnerProductAdd
{
    /// Boolean inner product.
    #[allow(clippy::arithmetic_side_effects)] // Used in unchecked mode
    fn inner_product<const CHECK: bool>(
        _cfg: &C,
        lhs: &[Boolean],
        rhs: &[Rhs],
        zero: Out,
    ) -> Result<Out, InnerProductError> {
        if lhs.len() != rhs.as_ref().len() {
            return Err(InnerProductError::LengthMismatch {
                lhs: lhs.len(),
                rhs: rhs.as_ref().len(),
            });
        }

        (0..lhs.len())
            .filter(|&i| lhs[i].into_inner())
            .try_fold(zero, |acc, i| {
                let rhs = Out::from_ref(&rhs[i]);
                if CHECK {
                    acc.checked_add(&rhs).ok_or(InnerProductError::Overflow)
                } else {
                    Ok(acc + rhs)
                }
            })
    }
}

#[cfg(test)]
mod test {
    use crate::{CHECKED, UNCHECKED};
    use crypto_bigint::{U64, Word, const_monty_params};
    use crypto_primitives::{
        FixedConfig, LiftElementWithConfig, crypto_bigint_const_monty::ConstMontyField,
        crypto_bigint_monty::MontyField, crypto_bigint_uint::Uint,
    };
    use num_traits::ConstZero;
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn test_inner_product_basic() {
        let lhs = [1, 2, 3];
        let rhs = [4, 5, 6];
        assert_eq!(
            MBSInnerProduct::inner_product::<CHECKED>(&(), &lhs, &rhs, 0),
            Ok(4 + 2 * 5 + 3 * 6)
        );
    }

    #[test]
    fn scalar_product() {
        let lhs = 42i32;
        let rhs = 23i128;

        assert_eq!(
            ScalarProduct::inner_product::<CHECKED>(&(), &lhs, &[rhs], 0).unwrap(),
            i128::from(lhs) * rhs
        )
    }

    #[test]
    fn boolean_checked_eq_mbs_inner_product() {
        let lhs = [
            Boolean::from(true),
            Boolean::from(false),
            Boolean::from(true),
            Boolean::from(true),
        ];
        let rhs = [1i128, 2, 3, 4];

        assert_eq!(
            BooleanInnerProductAdd::inner_product::<CHECKED>(&(), &lhs, &rhs, 0),
            MBSInnerProduct::inner_product::<CHECKED>(&(), &rhs, &lhs, 0i128)
        );
    }

    const_monty_params!(Params, U64, "0000000000000007");

    #[test]
    fn boolean_unchecked_eq_boolean_checked() {
        let lhs = [
            Boolean::from(true),
            Boolean::from(false),
            Boolean::from(true),
            Boolean::from(true),
        ];
        let rhs = [
            ConstMontyField::<Params, 1>::from(1),
            ConstMontyField::<Params, 1>::from(2),
            ConstMontyField::<Params, 1>::from(3),
            ConstMontyField::<Params, 1>::from(4),
        ];

        assert_eq!(
            BooleanInnerProductAdd::inner_product::<CHECKED>(
                &(),
                &lhs,
                &rhs,
                ConstMontyField::ZERO
            ),
            BooleanInnerProductAdd::inner_product::<UNCHECKED>(
                &(),
                &lhs,
                &rhs,
                ConstMontyField::ZERO
            )
        );
    }

    #[test]
    fn prepared_fixed_field_inner_product_retains_projection_semantics() {
        type StaticField = ConstMontyField<Params, 1>;

        let cfg = FixedConfig::<StaticField>::default();
        let lhs = [-5_i32, 11];
        let rhs = [StaticField::from(3_u64), StaticField::from(4_u64)];
        let expected =
            FieldInnerProduct::inner_product::<UNCHECKED>(&cfg, &rhs, &lhs, cfg.zero()).unwrap();

        assert_eq!(cfg.prepare_field_inner_product(&rhs)(&lhs), Ok(expected));
    }

    const FIELD_LIMBS: usize = 1;
    const INT_LIMBS: usize = 2;
    type DynamicField = MontyField<FIELD_LIMBS>;

    fn dynamic_field(modulus: Word) -> DynamicField {
        DynamicField::new(&Uint::new(CBUint::from(modulus))).expect("valid odd modulus")
    }

    #[test]
    fn prepared_montgomery_inner_product_matches_manual_modular_sum() {
        let cfg = dynamic_field(7);
        let lhs = [
            Int::from_words([0, Word::from(65_536_u32)]),
            Int::from(-5_i32),
            Int::from(11_i32),
        ];
        let rhs = [2_u64, 3, 4].map(|value| cfg.project(&value));
        let inner_product = cfg.prepare_field_inner_product(&rhs);

        let actual = inner_product(&lhs).expect("matching lengths");

        assert_eq!(cfg.lift(&actual), Uint::from(2_u64));
    }

    #[test]
    fn prepared_montgomery_inner_product_matches_projection_at_boundaries() {
        let cfg = dynamic_field(0xffff_ffff_ffff_ffc5);
        let lhs = [
            Int::<INT_LIMBS>::ZERO,
            Int::MIN,
            Int::from_words([0xffff_ffff_ffff_ffc5, 0]),
            -Int::from_words([0xdead_beef_cafe_babe, 1]),
            Int::from_words([0x8000_0000_0000_0041, 2]),
        ];
        let rhs = [
            0_u64,
            3,
            0xffff_ffff_ffff_ffc4,
            0xdead_beef_cafe_babe,
            0x8000_0000_0000_0041,
        ]
        .map(|value| cfg.project(&value));
        let expected =
            FieldInnerProduct::inner_product::<UNCHECKED>(&cfg, &rhs, &lhs, cfg.zero()).unwrap();
        let inner_product = cfg.prepare_field_inner_product(&rhs);

        assert_eq!(inner_product(&lhs), Ok(expected));
    }

    #[test]
    fn prepared_montgomery_inner_product_matches_projection_for_narrower_integers() {
        let cfg = MontyField::<2>::new(&Uint::new(CBUint::from(7_u64))).expect("valid odd modulus");
        let lhs = [Int::<1>::from(-5_i32), Int::from(11_i32)];
        let rhs = [3_u64, 4].map(|value| cfg.project(&value));
        let expected =
            FieldInnerProduct::inner_product::<UNCHECKED>(&cfg, &rhs, &lhs, cfg.zero()).unwrap();

        assert_eq!(cfg.prepare_field_inner_product(&rhs)(&lhs), Ok(expected));
    }

    #[test]
    fn prepared_montgomery_inner_product_rejects_length_mismatch() {
        let cfg = dynamic_field(7);
        let rhs = [cfg.one()];
        let inner_product = <DynamicField as PreparedFieldInnerProduct<
            Int<INT_LIMBS>,
        >>::prepare_field_inner_product(&cfg, &rhs);

        assert_eq!(
            inner_product(&[]),
            Err(InnerProductError::LengthMismatch { lhs: 0, rhs: 1 })
        );
    }

    proptest! {
        #[test]
        #[cfg_attr(miri, ignore)]
        fn prepared_montgomery_inner_product_matches_projection_for_dense_signed_inputs(
            entries in prop::collection::vec(
                (any::<Word>(), any::<Word>(), any::<Word>()),
                0..64,
            )
        ) {
            let cfg = dynamic_field(0xffff_ffff_ffff_ffc5);
            let lhs: Vec<_> = entries
                .iter()
                .map(|(lo, hi, _)| Int::from_words([*lo, *hi]))
                .collect();
            let rhs: Vec<_> = entries
                .iter()
                .map(|(_, _, value)| cfg.project(value))
                .collect();
            let expected =
                FieldInnerProduct::inner_product::<UNCHECKED>(&cfg, &rhs, &lhs, cfg.zero())
                    .unwrap();
            let inner_product = cfg.prepare_field_inner_product(&rhs);

            prop_assert_eq!(inner_product(&lhs), Ok(expected));
        }
    }
}
