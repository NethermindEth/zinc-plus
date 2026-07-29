use crate::{from_ref::FromRef, mul_by_scalar::MulByScalar};
use crypto_primitives::{ProjectElementWithConfig, SemiringConfig, Wrapper, boolean::Boolean};
use num_traits::CheckedAdd;
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
    use crypto_bigint::{U64, const_monty_params};
    use crypto_primitives::crypto_bigint_const_monty::ConstMontyField;
    use num_traits::ConstZero;

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
}
